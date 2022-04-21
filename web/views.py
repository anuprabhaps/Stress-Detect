from django.shortcuts import render, redirect, HttpResponseRedirect
from .models import Member
import random
from django.shortcuts import render
from django.http import HttpResponse
import joblib
import numpy as np
# Create your views here.

def details(request):
    return render(request, 'web/details.html')

def register(request):
    return render(request, 'web/register.html')

def login(request):
    return render(request, 'web/login.html')
def Convert(string_series):
    floats_list = [float(string_series.split())]
    return floats_list
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi

    while e > 0:
        temp1 = temp_phi//e
        temp2 = temp_phi - temp1 * e
        temp_phi = e
        e = temp2

        x = x2 - temp1 * x1
        y = d - temp1 * y1

        x2 = x1
        x1 = x
        d = y1
        y1 = y

    if temp_phi == 1:
        return d + phi

def is_prime(num):
    if num == 2:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num**0.5)+2, 2):
        if num % n == 0:
            return False
    return True

def generate_key_pair(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError('Both numbers must be prime.')
    elif p == q:
        raise ValueError('p and q cannot be equal')
    n = p * q

    phi = (p-1) * (q-1)
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)
    d = multiplicative_inverse(e, phi)
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    key, n = pk
    cipher = [pow(ord(char), key, n) for char in plaintext]
    return cipher

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

    msgtoencrypt= str([lis])
    print(msgtoencrypt)
   
    p=19
    q=23
    public,private = generate_key_pair(p,q)
    encrypted_msg = encrypt(public, msgtoencrypt)
    #decrypted_msg = decrypt(private,encrypted_msg)
    print(encrypted_msg)
    #print(decrypted_msg)
    #final_msg = Convert(decrypted_msg)
    ans = model.predict([lis])
    return render(request,'web/predict.html',{'ans':ans,'lis':lis,'encrypted_msg':encrypted_msg})
  
