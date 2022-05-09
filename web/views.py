from django.shortcuts import render, redirect, HttpResponseRedirect
from .models import Member
import random
from django.shortcuts import render
from django.http import HttpResponse
import joblib

# Create your views here.

def details(request):
    return render(request, 'web/details.html')

def register(request):
    return render(request, 'web/register.html')

def predict(request):
    model= joblib.load('C:/Users/anuraj/Documents/Healthcare/finalized-model.sav')
    lis=[]
    lis.append(request.POST.get('sr'))
    lis.append(request.POST.get('rr'))
    lis.append(request.POST.get('t'))
    lis.append(request.POST.get('lm'))
    lis.append(request.POST.get('bo'))
    lis.append(request.POST.get('rem'))
    lis.append(request.POST.get('sr.1'))
    lis.append(request.POST.get('hr'))

    msg= str([lis])
    print(msg)
   
    ans = model.predict([lis])
    return render(request,'web/predict.html',{'ans':ans,'lis':lis})
  
