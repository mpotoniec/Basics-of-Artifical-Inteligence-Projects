from  numpy import mean, max, min
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_percentage_value(pulsars, no_pulsars, number_data = False):
    no_pulsars_length = len(no_pulsars)
    pulsars_length = len(pulsars)

    percent_value = int((pulsars_length * 100) / no_pulsars_length)

    to_print = str('Pulsary stanowią ' + str(percent_value) + "% zestawu danych")

    if number_data == True:
        return percent_value
    else:
        return to_print

def summary_pulsars(data):

    results = data['target_class']
    pulsars = []
    no_pulsars = []
    for record in results:
        if record == 1: pulsars.append(record)
        else: no_pulsars.append(record)

    print('Długość zestawu danych =', len(results))
    print('Srednia wyników =', mean(results))
    print('maksymalna wartość wyniku =', max(results))
    print('minimalna wartość wyniku =', min(results))

    print(calculate_percentage_value(pulsars, no_pulsars))

    sns.countplot(x='target_class', data=data)
    plt.title('Wykres sygnałów radiowych 1 dla pulsarów i 0 dla sygnaów które pulsarami nie są')
    plt.xlabel('')
    plt.ylabel('Liczba wszystkich wyników')
    plt.show()

    return 0

def print_acc_loss_plot(hist, number_of_epoches):

    array = [i for i in range(1, number_of_epoches + 1)]

    plt.title('Wykres skuteczności przewidywań oraz błędu względem epok')
    plt.xlabel('Liczba epok')
    plt.ylabel('Skuteczność przewidywań oraz wartość błędu')

    plt.plot(array, hist.history['accuracy'], 'o--')
    plt.plot(array, hist.history['loss'], 'o--')

    plt.show()

def print_plots(hist, number_of_epoches):

    array = [i for i in range(1, number_of_epoches + 1)]

    plt.subplot(2,1,1)
    plt.title('Wykres skuteczności przewidywań od epok')
    plt.ylabel('Procent skuteczności przewidywań[%]')
    plt.plot(array, hist.history['accuracy'], 'o--')

    plt.subplot(2,1,2)
    plt.title('Wykres błędu przewidywań od epok')
    plt.ylabel('Błąd przewidywań')
    plt.plot(array, hist.history['loss'], 'o--')

    plt.show()

def calculate_percent(value):

    value = value * 100
    percent_value = int(value)
    value = value - percent_value
    
    value = value * 10
    number_after_decimal_point = int(value)
    value = value - number_after_decimal_point

    if value > 0.5: number_after_decimal_point += 1

    if number_after_decimal_point >= 9:
        percent_value += 1
    
    elif number_after_decimal_point <= 1:
        pass

    else:
        number_after_decimal_point = number_after_decimal_point / 10
        percent_value = float(percent_value)
        percent_value = percent_value + number_after_decimal_point

    output_string = str(percent_value) + '%'

    return output_string

def main():

    data = pd.read_csv('Dataset/pulsar_stars.csv')

    print('')
    summary_pulsars(data)
    print('')

    d = pd.get_dummies(data['target_class'])
    data = pd.concat([data,d],axis = 1)
    data.drop('target_class',axis = 1, inplace = True)

    x_train,x_test,y_train,y_test=train_test_split(data.drop([0,1],axis=1),data[[0,1]],test_size=0.25)

    model = Sequential()

    model.add(Dense(64, input_shape=(8,), activation = 'relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(128, activation = 'relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax')) 


    model.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy']) 

    print('Trenowanie modelu...')
    epoches = 10  
    hist = model.fit(x_train, y_train, epochs = epoches)

    print('Podsumowanie modelu: ')
    model.summary()

    print("Testowanie modelu:")
    loss_value, acc_value = model.evaluate(x_test, y_test) 
    print("Accurancy =", calculate_percent(acc_value), '(', acc_value, ')') 
    print('Loss', loss_value)

    print_plots(hist, epoches)

    return 0


if __name__ == "__main__":
    print('Program was executed with code:', main())