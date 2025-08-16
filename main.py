

import torch
from pathlib import Path
from src.preprocessig import read_data_normalization, make_sequencer_and_dataloader
from src.train_phase import train_model
from src.test_phase import evaluate
from src.plotting_Curves  import plot_predictions
from src.LSTM_model import model_LSTM



def main():


    Path = '/home/jobe/Desktop/my_venv/working/lstm-jena_code/jena_dataset/jena_climate_2009_2016.csv'
    # get data and normalization
    raw_data, temperature, num_train_samples, num_val_samples, num_test_samples = read_data_normalization(path_to_jena_Dataset= Path, train_samples_percent= 50, val_samples_percent = 25)

    


    train_dataloader = make_sequencer_and_dataloader(raw_data[0:num_train_samples , :],temperature[:num_train_samples], \
                                                     
                                                     seq_length = 120, pred_length = 24, sampling_rate = 6,batch_size = 512, shuffle = False)
    



    validation_dataloader = make_sequencer_and_dataloader(raw_data[num_train_samples:(num_val_samples+num_train_samples),:],temperature[num_train_samples:(num_val_samples+num_train_samples)], \
                                                     
                                                     seq_length = 120, pred_length = 24, sampling_rate = 6,batch_size = 512, shuffle = False)




    test_dataloader = make_sequencer_and_dataloader(raw_data[(num_val_samples+num_train_samples):,:],temperature[(num_val_samples+num_train_samples):], \
                                                     
                                                     seq_length = 120, pred_length = 24, sampling_rate = 6,batch_size = 512, shuffle = False)



    # -------- MODEL LSTM ---------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    input_size = 14
    hiden_size = 32
    model = model_LSTM(input_size = input_size, hidden_size=hiden_size , batch_first=True,  num_layers=2)

    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)



    # ----------- Training -----------------
    model = train_model(model, train_dataloader, validation_dataloader, optimizer, criterion, device)
    
    save_model_path = '/home/jobe/Desktop/my_venv/working/lstm-jena_code/save_models/lstm_model_epoch_50.pth'
    torch.save(model.state_dict(), save_model_path)


    # ----------- Evaluation ----------------
    predictions, truths = evaluate(model, test_dataloader, device )


    # ----------- Plotting ----------------
    
    plot_predictions(truths, predictions)




if __name__ == "__main__":

    main()