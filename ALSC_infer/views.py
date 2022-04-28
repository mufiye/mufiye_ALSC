from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse

# try for infer.py infer function
import torch
import torch.nn.functional as F
import numpy as np
from infer import *

inf = Inferer()
def index(request):
    if request.method == 'POST':
        sentence = request.POST.get('sentence')
        aspect = request.POST.get('aspect')
        # result = infering(sentence,aspect) #还有三个选项,关于网页的就两个
        model_name = request.POST.get('model')
        dataset_name = request.POST.get('dataset')
        # print(model_choice)
        # print(dataset_choice)
        if model_name == 'non-BERT':
            model_choice = 1
        else:
            model_choice = 2
        print("model_choice: {}".format(model_choice))
        if dataset_name == 'rest':
            dataset_choice = 1
        elif dataset_name == 'laptop':
            dataset_choice = 2
        else:
            dataset_choice = 3
        print("dataset_choice: {}".format(dataset_choice))
        resultNum = inf.evaluate(sentence, aspect, model_choice, dataset_choice)
        print("result number: {}".format(resultNum))
        if resultNum == 0:
            result = "the sentiment polarity of {} is negative".format(aspect)
        elif resultNum == 1:
            result = "the sentiment polarity of {} is positive".format(aspect)
        else:
            result = "the sentiment polarity of {} is neutral".format(aspect)
        return render(request, 'ALSC_infer/index.html',{'sentence':sentence,'aspect':aspect,'result':result,'model_choice':model_choice,"dataset_choice":dataset_choice})
    return render(request, 'ALSC_infer/index.html',{'model_choice':1,"dataset_choice":1})

# the query input page about ALSC infering
# def query(request):
#     # result = infering()
#     try:
#         text = request.POST['text']
#         aspect = request.POST['aspect']
#         print(text)
#         print(aspect)
#         result = infering(text,aspect)
#     except:
#         return render(request, 'ALSC_infer/query.html')
#     else:
#         return HttpResponseRedirect(reverse('showResult', args=result))
def query(request):
    # result = infering()
    return render(request, 'ALSC_infer/query.html')

# the result output page about ALSC infering
def showResult(request):
    text = request.POST['text']
    aspect = request.POST['aspect']
    print(text)
    print(aspect)
    result = infering(text,aspect)
    return render(request, 'ALSC_infer/showResult.html', {'result': result})

    
