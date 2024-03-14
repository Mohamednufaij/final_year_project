from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from pyresparser import ResumeParser
from .models import InterviewData, UserLogin
from django.contrib import messages
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import os
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import tweepy
import pickle
from django.shortcuts import render, get_object_or_404



# API keys from developer.twitter.com
apikey = 'CqvNMKklaTTsWyIm6tzoVi2lE'
apisecretkey = 'Q8FT3FAexZ9nXLaZIGyF4nhR5i0b2uze6S0vjsMvsP63A8HCj6'
acctoken = '1394557783853338625-ltE1CDpOSboREb9SD9lGwyB8OXTDDy'
acctokensecret = 'dCz2WtIFdCNZdahJOvS1SKYBhKDVzjdCGP6OAEO3VZldP'

p = PorterStemmer()
stopwords = stopwords.words("english")


def filtr(text):
    # Remove URLs, mentions, and hashtags
    filtered_text = re.sub(r"http\S+|@\S+|#\S+", "", text)

    # Tokenize, remove stopwords, and apply stemming
    tokens = filtered_text.split()
    filtered_tokens = [p.stem(token) for token in tokens if token not in stopwords]

    return ' '.join(filtered_tokens)


# Create your views here.

def index(request):
    return render(request, 'index.html')


def r_login(request):
    return render(request, 'Rlogin.html')


def interview(request):
    firstname = request.POST['fname']
    lastname = request.POST['lastname']
    email = request.POST['email']
    mobile_number = request.POST['mob']

    applicants = UserLogin.objects.all()
    for applicant in applicants:
        if applicant.email == email:
            messages.error(request, 'Email ID already exists!')
            return redirect(index)
        elif applicant.phone == mobile_number:
            messages.error(request, 'Phone number already exists!')
            return redirect(index)

    userlogin = UserLogin(
        fname=firstname,
        lname=lastname,
        email=email,
        phone=mobile_number,
    )
    userlogin.save()

    return render(request, 'interview.html')

#To navigate interview page directly
def interviewDirect(request):
    return render(request, 'interview.html')


def success(request):
    firstname = request.POST.get('name')
    email = request.POST.get('email')
    phone = request.POST.get('phone')
    # website = request.POST.get('website')
    age = request.POST.get('age')
    gender = request.POST.get('gender')
    openness = request.POST.get('openness')
    neuroticism = request.POST.get('neuroticism')
    conscientiousness = request.POST.get('conscientiousness')
    agreeableness = request.POST.get('agreeableness')
    extraversion = request.POST.get('extraversion')
    resume = request.FILES.get('resume')

    # print(resume)
    # if resume:
    #     print('hello')

    # model prediction
    df = pd.read_csv(r'C:\\Users\\neera\\OneDrive\\Desktop\\mysite\\main\\trainDataset.csv')
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    x_train = df.iloc[:, :-1].to_numpy(dtype=str)
    y_train = df.iloc[:, -1].to_numpy(dtype=str)
    lreg = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
    lreg.fit(x_train, y_train)

    if gender == 'male':
        gender = 1
    elif gender == 'female':
        gender = 0
    # print(gender)
    # input_data = [[gender, int(age), int(openness), int(neuroticism), int(conscientiousness), int(agreeableness),
    #                int(extraversion)]]

    # print(input_data)
    # print('*' * 100)
    # predi = str(lreg.predict(input_data)[0]).capitalize()
    # print(pred)
    # print('*' * 100)

    # auth = tweepy.OAuthHandler(apikey, apisecretkey)
    # auth.set_access_token(acctoken, acctokensecret)
    # api = tweepy.API(auth)
    # user = website

    # Retrieve user's last tweets
    # tweets = api.user_timeline(screen_name=user, count=50)
    # text = []

    # for tweet in tweets:
    #     filtered_tweet = filtr(tweet.text)
    #     text.append(filtered_tweet)
    #     text.append("|||")
    #
    # text = ' '.join(text)

    # Load the pre-trained model and count vectorizer
    tcv = CountVectorizer(decode_error="replace", vocabulary=pickle.load(
        open("C:\\Users\\neera\\OneDrive\\Desktop\\mysite\\main\\vocab.pkl", "rb")))
    model = load_model('C:\\Users\\neera\\OneDrive\\Desktop\\mysite\\main\\mymodel.h5')

    # Predict the personality type
    # predictions = model.predict(tcv.transform([text]))
    # res = np.argmax(predictions)
    # personality_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
    #                      'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    # predicted_personality = personality_types[res]

    fs = FileSystemStorage()
    fs.save(resume.name, resume)

    # Parsing resume data
    resume_data = ResumeParser(fs.path(resume.name)).get_extracted_data()
    print(resume_data)

    json_data = json.dumps(resume_data)
    resume_folder = 'resume'
    os.makedirs(resume_folder, exist_ok=True)  # Create the folder if it doesn't exist
    resume_file_path = os.path.join(resume_folder, resume.name + '.json')
    with open(resume_file_path, 'w') as file:
        file.write(json_data)

    # Creating InterviewData object and saving it to the database
    applicant = InterviewData(
        firstname=firstname,
        email=email,
        phone=phone,
        # website=website,
        age=age,
        gender=gender,
        openness=openness,
        neuroticism=neuroticism,
        conscientiousness=conscientiousness,
        agreeableness=agreeableness,
        extraversion=extraversion,
        # resume=resume.name,
        pred=predi,
        # twitterper=predicted_personality
    )
    applicant.save()

    return render(request, 'success.html', {'name': firstname})


def candidates_sub(request):
    remail = request.POST['email']
    rpass = request.POST['pass']

    with open('C:\\Users\\neera\\OneDrive\\Desktop\\mysite\\credentials.txt', 'r') as companymail:
        data = companymail.readlines()

    if data[0].strip('\n') == remail:
        if data[1] == rpass:

            candidates = InterviewData.objects.all()

            return render(request, 'r1.html', {'candidates': candidates})
        else:
            messages.error(request, 'Wrong Password!')
            return redirect(r_login)
    else:
        messages.error(request, 'Wrong Email!')
        return redirect(r_login)


def view_candidate_result(request, candidate_id):
    candidate = get_object_or_404(InterviewData, id=candidate_id)
    # print(candidate.resume.name)

    with open(
            f'C:\\Users\\neera\\OneDrive\\Desktop\\mysite\\resume\\{candidate.resume.name}.json') as file:
        data = json.load(file)

    skills = data['skills']
    degrees = data['degree']
    # print(degrees)

    with open(f'C:\\Users\\neera\\OneDrive\\Desktop\\mysite\\desc\\{candidate.pred}.txt', 'r') as pers1:
        data2 = pers1.readlines()

    # with open(f'C:\\Users\\neera\\OneDrive\\Desktop\\mysite\\desc\\{candidate.twitterper}.txt', 'r') as pers2:
    #     data3 = pers2.readlines()

    context = {
        'candidate': candidate,
        'skills': skills,
        'degree': degrees,
        'pred1': data2,
        # 'pred2': data3,
    }

    return render(request, 'candidatedetails.html', context)
