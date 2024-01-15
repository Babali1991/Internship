#!/usr/bin/env python
# coding: utf-8

# In[1]:


#python program to print factors of a number


# In[6]:


n = int(input("Enter a number"))
for x in range(1,n+1):
    if n%x ==0:
        print(x)
    


# In[7]:


#find python program whether pailandrome or not


# In[3]:


Orginal = input("Enter a string:")
Reverse = Orginal [::-1]
if Orginal == Reverse:
    print("string is pailndrome...")
else:     
          
    print("string is not pailndrome...")


# In[4]:


#python program to print the frequency of each character present in a given string.


# In[6]:


str = input("Enter String")
#print(str)
l=list(str)
#print(l)
freq=[l.count(ele) for ele in l]
#print(freq)
d=dict(zip(l,freq))
print(d)


# In[7]:


#python program to find number is prime or not.


# In[8]:


num = int(input("Enter the num :"))
count = 0
i=1
while i<=num:
    if num%1==0:
        count=count+1
        i=i+1
        
    if count==2:
        print("it is a prime num:")
    elif count>2:
        print("it is a composite num")
    else:
        print("the number is neither prime nor composite")
        

        
        

        


# In[9]:


num = int(input("Enter the num :"))
count = 0
i=1
while i<=num:
    if num%1==0:
        count=count+1
        i=i+1
        
    if count==2:
        print("it is a prime num:")
    elif count>2:
        print("it is a composite num")
    else:
        print("the number is neither prime nor composite")
        


# In[ ]:




