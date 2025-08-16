
import matplotlib.pyplot as plt
import os

def plot_predictions(true, pred ):
    

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(true[-24:])),   true[-24:],  'b'  ,    label='Real Temperatures /degC',    alpha=0.7)
    plt.plot(range(len(pred[-24:])),   pred[-24:],  'r--' ,  label='Predicted Temperature /degC',  alpha=0.7)

    plt.title("One Day Temperature Prediction( 24 hours)")
    plt.xlabel("Hours")
    plt.ylabel("Temperature /degC")
    plt.legend()
    # plt.show()


   
    image_dir = "/home/jobe/Desktop/my_venv/working/lstm-jena_code/imgs"
    save_path = os.path.join(image_dir, "temperature_prediction_50_epoch.png")

    # Save the image
    plt.savefig(save_path, dpi=300)  

    plt.close()  