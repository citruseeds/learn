# Python
## Output
print() function:

``` py
# Prints "Hello World"
print("Hello World")

# Prints "Hello World" as well
print("Hello", "World")  
# Connecting strings with print() isn't done with +, just input the variables as parameters;
# spaces are also automatically appended whenever this is done, excluding the final parameter
```

print() has parameters `sep`, `end`, `file`, `flush`;

* `sep` is the character that will separate the concatenated parameters.
* `end` is the character that will be appended at the end of the print.
* `file` ?
* `flush` ?

To use any of these, the parameter must be explicitly specified to differentiate it from string concatenation arguments:

``` py
# Prints "Hello@World#"
print("Hello", "World", sep="@", end="#") 
```

https://docs.python.org/3/library/functions.html#print


## Comments
``` py
# Python comments.

"""
Multi-line comment.
Uses 3 double quotes (") or single quotes (').

These are supposed to be string literals, but Python ignores them if they're
not assigned to any variable, so it can function as a comment.
"""

'''
This also works fine.
This style of multi-line comments apparently isn't favored.
'''

# Multi-line comments are apparently preferably
# done using multiple single-line comments.
```

### Docstrings

## References/Further Reading
* [Learn X in Y minutes, Where X=Python](https://learnxinyminutes.com/docs/python/)
* [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
* [Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)

## Unsorted
* docstrings (triple " usage)