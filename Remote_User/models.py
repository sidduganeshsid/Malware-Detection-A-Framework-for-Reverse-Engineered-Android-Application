from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=300)
    gender= models.CharField(max_length=30)

class app_type_detection(models.Model):


    App_Name= models.CharField(max_length=3000)
    Category= models.CharField(max_length=3000)
    Reviews= models.CharField(max_length=3000)
    Size= models.CharField(max_length=3000)
    Installs= models.CharField(max_length=3000)
    Type= models.CharField(max_length=3000)
    Price= models.CharField(max_length=3000)
    Content_Rating= models.CharField(max_length=3000)
    Genres= models.CharField(max_length=3000)
    Last_Updated= models.CharField(max_length=3000)
    Current_Ver= models.CharField(max_length=3000)
    Android_Ver= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



