
from django.shortcuts import render, redirect, get_object_or_404

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
#model selection
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix, classification_report
# Create your views here.
from Remote_User.models import ClientRegister_Model,app_type_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Malware_Prediction(request):
    if request.method == "POST":
        if request.method == "POST":

            App_Name= request.POST.get('App_Name')
            Category= request.POST.get('Category')
            Reviews= request.POST.get('Reviews')
            Size= request.POST.get('Size')
            Installs= request.POST.get('Installs')
            Type= request.POST.get('Type')
            Price= request.POST.get('Price')
            Content_Rating= request.POST.get('Content_Rating')
            Genres= request.POST.get('Genres')
            Last_Updated= request.POST.get('Last_Updated')
            Current_Ver= request.POST.get('Current_Ver')
            Android_Ver= request.POST.get('Android_Ver')

        data = pd.read_csv("Android_Apps_Datasets.csv", encoding='latin-1')

        mapping = {'No Malware': 0, 'Malware Detected': 1}

        data['Label'] = data['Malware_Status'].map(mapping)

        x = data['App']
        y = data['Label']

        #cv = CountVectorizer()

        print(x)
        print("Y")
        print(y)

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
        #x = cv.fit_transform(data['App'].apply(lambda x: np.str_(x)))

        x = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))

        print("Random Forest Classifier")
        from sklearn.ensemble import RandomForestClassifier
        RFC = RandomForestClassifier(random_state=0)
        RFC.fit(X_train, y_train)
        pred_rfc = RFC.predict(X_test)
        RFC.score(X_test, y_test)
        print("ACCURACY")
        print(accuracy_score(y_test, pred_rfc) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, pred_rfc))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, pred_rfc))
        models.append(('RFC', RFC))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        App_Name1 = [App_Name]
        vector1 = cv.transform(App_Name1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = str(pred.replace("]", ""))

        prediction = int(pred1)

        if prediction == 0:
            val = 'No Malware'
        elif prediction == 1:
            val = 'Malware Detected'

        print(prediction)
        print(val)

        app_type_detection.objects.create(
        App_Name=App_Name,
        Category=Category,
        Reviews=Reviews,
        Size=Size,
        Installs=Installs,
        Type=Type,
        Price=Price,
        Content_Rating=Content_Rating,
        Genres=Genres,
        Last_Updated=Last_Updated,
        Current_Ver=Current_Ver,
        Android_Ver=Android_Ver,
        Prediction=val)

        return render(request, 'RUser/Predict_Malware_Prediction.html',{'objs': val})
    return render(request, 'RUser/Predict_Malware_Prediction.html')



