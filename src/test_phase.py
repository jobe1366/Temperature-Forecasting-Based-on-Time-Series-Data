import torch
import numpy as np


def evaluate(model, loader, device ):

    path_to_saved_model = '/home/jobe/Desktop/my_venv/working/lstm-jena_code/save_models/lstm_model_epoch_50.pth'
    model.load_state_dict(torch.load(path_to_saved_model, weights_only=True))

    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for X, y in loader:
            
            X, y = X.to(device), y.to(device)
            
            
            preds = model(X).cpu().numpy()
            y = y.cpu().numpy()  
            
            
            predictions.extend(preds.tolist())
            truths.extend(y.tolist())
    
    return np.array(predictions), np.array(truths)

if __name__ == "__main__":

    import numpy as np 
    import torch
    from pathlib import Path
    from preprocessig import read_data_normalization, make_sequencer_and_dataloader
    from plotting_Curves  import plot_predictions
    from LSTM_model import model_LSTM

    Path = '/home/jobe/Desktop/my_venv/working/lstm-jena_code/jena_dataset/jena_climate_2009_2016.csv'

    raw_data, temperature, num_train_samples, num_val_samples, num_test_samples = read_data_normalization(path_to_jena_Dataset= Path, train_samples_percent= 50, val_samples_percent = 25)

    test_dataloader = make_sequencer_and_dataloader(raw_data[(num_val_samples+num_train_samples):,:],temperature[(num_val_samples+num_train_samples):], \
                                                     
                                                     seq_length = 120, pred_length = 24, sampling_rate = 6, batch_size = 512, shuffle = False)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    input_size = 14
    hiden_size = 32
    model = model_LSTM(input_size = input_size, hidden_size=hiden_size , batch_first=True,  num_layers=2)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
    

    predictions, truths = evaluate(model, test_dataloader, device )
    plot_predictions(truths, predictions)

