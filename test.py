#!/usr/bin/python3

for letter in 'Python':     # First Example
   if letter == 'h':
      break
   print('Current Letter :', letter)

def breaker():
    if letter == 'h':
        raise Exception

for letter in 'Python':     # First Example
   try:
       breaker()
   except Exception:
       break
   finally:
       print('Current Letter :', letter)