import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PredNet (nn.Module):
    def __init__(self, stack_sizes, R_stack_sizes, 
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1.0, error_activation='relu', A_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid', output_mode='prediction', extrap_start_time=None):
        super(PredNet, self).__init__()

        self.stack_sizes=stack_sizes
        self.nb_layers=len(stack_sizes) # number of layers (l)
        self.R_stack_sizes=R_stack_sizes
        self.A_filt_sizes=A_filt_sizes # E --A filter--> A (target)
        self.Ahat_filt_sizes=Ahat_filt_sizes # R --Ahat filter--> Ahat (prediction)
        self.R_filt_sizes = R_filt_sizes # filters inside R (convLSTM)
        self.pixel_max = pixel_max # max number of pixels in output
        self.output_mode = output_mode
        self.extrap_start_time = extrap_start_time

        # validate the numbers
        assert len(R_stack_sizes) == self.nb_layers
        assert len(A_filt_sizes) == (self.nb_layers - 1) # since the top does not have a layer above
        assert len(Ahat_filt_sizes) == self.nb_layers # every layer makes a prediction
        assert len(R_filt_sizes) == self.nb_layers # each R, thus no of layers

        self.error_activation = self._get_activation(error_activation)
        self.A_activation = self._get_activation(A_activation)
        self.LSTM_activation = self._get_activation(LSTM_activation)

        if LSTM_inner_activation == 'hard_sigmoid':
            self.LSTM_inner_activation = self.hard_sigmoid # following the og implementation
        else:
            self.LSTM_inner_activation = self._get_activation(LSTM_inner_activation)


        # conv_layers: dictionary holding all trainable convolution kernels
        # Keys map to specific operations in the PredNet architecture:
        #   'i', 'f', 'o', 'c' : ConvLSTM internals (Input, Forget, Output gates, Cell update).
        #   'ahat'             : Prediction (projects R state -> Prediction Ahat).
        #   'a'                : Target processing (projects Error E below-> Target A for next layer).
        self.conv_layers = nn.ModuleDict()
        # LSTM gate conv
        for c in ['i','f','c','o']: # for each gate (no of channels)
            self.conv_layers[c]=nn.ModuleList()
            for l in range (self.nb_layers):
                # input of R = r_tm1[l]  +  e_tm1[l] (multiply 2 since errors are split into + and -)
                in_channels=R_stack_sizes[l] + 2*stack_sizes[l]
                if l < self.nb_layers-1: # excludes highest layer has no layers above
                    in_channels+=R_stack_sizes[l+1]
            
                self.conv_layers[c].append(
                    nn.Conv2d(in_channels,R_stack_sizes[l],R_filt_sizes[l],padding=R_filt_sizes[l]//2)
                ) # input_depth,output_depth (new state of LSTM),kernel size,padding (same padding preserve HxW)
        # ahat conv
        self.conv_layers['ahat']=nn.ModuleList()
        for l in range(self.nb_layers):
            self.conv_layers['ahat'].append(
                nn.Conv2d(R_stack_sizes[l],stack_sizes[l],Ahat_filt_sizes[l],padding=Ahat_filt_sizes[l]//2)
            ) # R (rep state) ----> prediction ahat
        # target conv
        self.conv_layers['a']=nn.ModuleList()
        for l in range (self.nb_layers-1):
            self.conv_layers['a'].append(
                nn.Conv2d(2*stack_sizes[l],stack_sizes[l+1],A_filt_sizes[l],padding=A_filt_sizes[l]//2)
            ) # error ----> target (layer ABOVE)
        
        self.upsample=nn.Upsample(scale_factor=2)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

    def _get_activation(self, name):
        if name == 'relu': return F.relu
        if name == 'tanh': return torch.tanh
        if name == 'sigmoid': return torch.sigmoid
        return F.relu # default
    
    def hard_sigmoid(self, x):
        return torch.clamp(0.2 * x + 0.5, 0., 1.)
    
    # layer structures (halve each time) more layers -> smaller at the top layers..
    def init_hidden(self, batch_size, input_shape, device='cpu'):
        init_states = {'r': [], 'c': [], 'e': []}
        
        current_h, current_w = input_shape[1], input_shape[2]
        
        for l in range(self.nb_layers):
            # (B, c, h, w)
            r_shape = (batch_size, self.R_stack_sizes[l], current_h, current_w)
            c_shape = (batch_size, self.R_stack_sizes[l], current_h, current_w) # long term memory
            e_shape = (batch_size, 2 * self.stack_sizes[l], current_h, current_w)
            
            init_states['r'].append(torch.zeros(r_shape).to(device))
            init_states['c'].append(torch.zeros(c_shape).to(device))
            init_states['e'].append(torch.zeros(e_shape).to(device))
            
            current_h //= 2
            current_w //= 2
            
        return init_states
    
    # one frame
    def step(self,a,states,t):

        r_tm1=states['r']
        c_tm1 = states['c']
        e_tm1 = states['e']

        if self.extrap_start_time is not None and t >= self.extrap_start_time:
            a = states.get('prev_prediction', a)
        r=[]
        c=[]
        e=[]

        # top down updating R
        r_up=None
        for l in reversed(range(self.nb_layers)):
            inputs=[r_tm1[l],e_tm1[l]]   
            if l<self.nb_layers-1:
                inputs.append(r_up)   
            inputs = torch.cat(inputs,dim=1) # along second dim (channels) 

            i = self.LSTM_inner_activation(self.conv_layers['i'][l](inputs))
            f = self.LSTM_inner_activation(self.conv_layers['f'][l](inputs))
            o = self.LSTM_inner_activation(self.conv_layers['o'][l](inputs))
            _c = f * c_tm1[l] + i * self.LSTM_activation(self.conv_layers['c'][l](inputs))
            _r = o * self.LSTM_activation(_c)
            
            c.insert(0, _c)
            r.insert(0, _r)
            
            if l > 0:
                r_up = self.upsample(_r)
        
        # bottom up update error
        frame_prediction = None
        
        for l in range(self.nb_layers):
            ahat=self.conv_layers['ahat'][l](r[l])
            if l==0:
                ahat=torch.min(ahat, torch.tensor(self.pixel_max).to(ahat.device)) # cap pixel values
                frame_prediction=ahat
            # computing e
            e_up=self.error_activation(ahat-a)
            e_down=self.error_activation(a-ahat)

            e_l=torch.cat((e_up,e_down), dim=1)
            e.append(e_l)

            if l < self.nb_layers - 1:
                a = self.conv_layers['a'][l](e_l) # conv
                a = self.A_activation(a) 
                a = self.pool(a) # pool

        # output mode
        if self.output_mode == 'prediction':
            step_output = frame_prediction
        elif self.output_mode == 'error':
            step_output = torch.stack([torch.mean(layer_e.view(layer_e.size(0), -1), dim=1) for layer_e in e], dim=1) # mean of score
        elif self.output_mode == 'all':
            flat_pred = frame_prediction.view(frame_prediction.size(0), -1)
            layer_errors = torch.stack([torch.mean(layer_e.view(layer_e.size(0), -1), dim=1) for layer_e in e], dim=1)
            step_output = torch.cat((flat_pred, layer_errors), dim=1) # [Pixel1, Pixel2, ..., PixelN, Error1, Error2, Error3]
        else:
            step_output = frame_prediction

        new_states = {'r': r, 'c': c, 'e': e}
        if self.extrap_start_time is not None:
            new_states['prev_prediction'] = frame_prediction
            
        return step_output, new_states
    
    def forward(self,sequences,initial_states=None):
        batch_size, time_steps, channels, height, width = sequences.size()
        
        if initial_states is None:
            states = self.init_hidden(batch_size, (channels, height, width), device=sequences.device)
        else:
            states = initial_states

        output_list = []
        
        for t in range(time_steps):
            a = sequences[:, t, :, :, :]
            step_output, states = self.step(a, states, t)
            output_list.append(step_output)
            
        return torch.stack(output_list, dim=1)

if __name__ == '__main__':
    # 1. Define Model Hyperparameters
    n_channels = 3       # e.g., RGB images
    img_height = 128
    img_width  = 160
    
    # Model Setup (3 Layers)
    # Layer 0: 3 input channels
    # Layer 1: 48 hidden channels
    # Layer 2: 96 hidden channels
    stack_sizes = (n_channels, 48, 96)
    R_stack_sizes = stack_sizes
    
    A_filt_sizes = (3, 3)          # Filter sizes for Target (A) modules
    Ahat_filt_sizes = (3, 3, 3)    # Filter sizes for Prediction (Ahat) modules
    R_filt_sizes = (3, 3, 3)       # Filter sizes for Representation (R/LSTM) modules
    
    # 2. Instantiate Model
    prednet = PredNet(
        stack_sizes=stack_sizes,
        R_stack_sizes=R_stack_sizes,
        A_filt_sizes=A_filt_sizes,
        Ahat_filt_sizes=Ahat_filt_sizes,
        R_filt_sizes=R_filt_sizes,
        pixel_max=1.0,
        error_activation='relu',
        A_activation='relu',
        LSTM_activation='tanh',
        LSTM_inner_activation='hard_sigmoid',
        output_mode='prediction',  # 'prediction', 'error', or 'all'
        extrap_start_time=None     # Set to an int (e.g., 5) to start extrapolating after frame 5
    )
    print(prednet)
    
    # 3. Create Dummy Input Data
    # Shape: (Batch Size, Time Steps, Channels, Height, Width)
    batch_size = 4
    time_steps = 10
    inputs = torch.randn(batch_size, time_steps, n_channels, img_height, img_width)
    
    print(f"Input shape: {inputs.shape}")

    # 4. Run Forward Pass
    with torch.no_grad():
        output = prednet(inputs)
        
    # 5. Check Outputs
    print(f"Output shape: {output.shape}")
    if prednet.output_mode == 'prediction':
        # (Batch, Time, Channels, Height, Width)
        assert output.shape == inputs.shape
        print("Success! Output shape matches input shape (Next Frame Prediction).")
    elif prednet.output_mode == 'error':
        # (Batch, Time, Num_Layers)
        expected_shape = (batch_size, time_steps, len(stack_sizes))
        assert output.shape == expected_shape
        print(f"Success! Output shape matches error shape: {expected_shape}")