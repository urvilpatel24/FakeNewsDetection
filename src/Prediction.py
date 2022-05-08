import pickle

class Prediction(object):

     def __init__(self, final_model):
        self.final_model = final_model
        

     #function to run for prediction
     def detecting_fake_news(self):  
     #retrieving the best model for prediction call

        self.var = input("Please enter the news text you want to verify: ")
        print("You entered: " + str(self.var))
      
        load_model = pickle.load(open(self.final_model, 'rb'))
        prediction = load_model.predict([self.var])
        prob = load_model.predict_proba([self.var])

        if prediction[0] == 0:
            print("Prediction of the News :  Looking Fake News")
        else:
            print("Prediction of the News : Looking Real News")

        return print("The truth probability score is ",prob[0][1])
