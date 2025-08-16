


import torch.nn as nn 


class model_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, batch_first = True, num_layers = 1):
		super().__init__()
		self.lstm = nn.LSTM(input_size=input_size, 
							hidden_size=hidden_size, 
							num_layers=num_layers, 
							batch_first=batch_first) 
		 
		self.dropout = nn.Dropout(p=0.3)

		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, x):
		out, (h,c) = self.lstm(x)
		dropout = self.dropout(out)
		out = self.fc(dropout[:,-1,:])

		return out
	



if __name__ == "__main__":
	
        import torch
        import torch.nn as nn 
        from torchsummary import summary
        

        input_size = 14
        hiden_size = 32
        model = model_LSTM(input_size = input_size, hidden_size=hiden_size , batch_first=True,  num_layers=2)
		
        summary(model, torch.zeros(1, 120, 14))
		
		