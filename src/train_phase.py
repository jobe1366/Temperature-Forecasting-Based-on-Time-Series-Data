
import torch


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=50):
    
    for epoch in range(epochs):

        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1,),  y_batch)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:

                X_val = X_val.to(device)
                y_val = y_val.to(device)

                outputs = model(X_val)
                val_loss += criterion(outputs.view(-1,),  y_val).item()
        


        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}")
        
    return model