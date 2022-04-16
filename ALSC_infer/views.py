from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse

# try for infer.py infer function
from infer import infering

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

    
