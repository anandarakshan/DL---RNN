# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

<img width="1018" height="336" alt="Screenshot 2026-05-19 091532" src="https://github.com/user-attachments/assets/0b0bae35-13ea-4414-8acf-3fcf4def1213" />


## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.


### STEP 3: 
Define the RNN model architecture.


### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 
Predict on test data, plot actual vs. predicted prices. 




## PROGRAM

### Name: Ananda Rakshan K V

### Register Number: 212223230014

```python
class RNNModel(nn.Module):
    # write your code here
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
      super(RNNModel,self).__init__()
      self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x):
      out,_=self.rnn(x)
      out=self.fc(out[:,-1,:])
      return out
# Train the Model
def train_model(model,train_loader,criterion,optimizer,epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch,y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
  # Plot training loss
  print('Name: Ananda Rakshan K V')
  print('Register Number: 212223230014')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_model(model,train_loader,criterion,optimizer)
```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="596" height="695" alt="image" src="https://github.com/user-attachments/assets/1138d80c-e508-40c1-bc71-9e5c13c7d3bc" />


## True Stock Price, Predicted Stock Price vs time

<img width="1037" height="581" alt="Screenshot 2026-05-19 093450" src="https://github.com/user-attachments/assets/534fdebf-f391-4e15-852e-5705032b3734" />


### Predictions
<img width="391" height="52" alt="Screenshot 2026-05-19 093509" src="https://github.com/user-attachments/assets/1d2e78d4-baba-44e5-ae14-f703e053a202" />



## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
