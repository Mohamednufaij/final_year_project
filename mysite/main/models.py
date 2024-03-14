from django.db import models

# Create your models here.


class UserLogin(models.Model):
    fname = models.CharField(max_length=100)
    lname = models.CharField(max_length=100)
    email = models.CharField(max_length=50)
    phone = models.CharField(max_length=100)


class InterviewData(models.Model):
    firstname = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    phone = models.CharField(max_length=100)
    website = models.CharField(max_length=100, blank=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female')])
    openness = models.IntegerField()
    neuroticism = models.IntegerField()
    conscientiousness = models.IntegerField()
    agreeableness = models.IntegerField()
    extraversion = models.IntegerField()
    resume = models.FileField(upload_to='resumes/')
    pred = models.CharField(max_length=100)
    twitterper = models.CharField(max_length=100)
