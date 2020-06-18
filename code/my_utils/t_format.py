from colorama import init
from termcolor import cprint, colored
init(autoreset=True)
p_color = "yellow"

def t_format(text, text_length=0):
    if text_length==0:
        return "%-10s" % text 
    if text_length==0.5:
        return "%-15s" % text 
    elif text_length==1:
        return "%-20s" % text
    elif text_length==2:
        return "%-25s" % text
    elif text_length==3:
        return "%-30s" % text
    elif text_length==4:
        return "%-40s" % text
    else:
        return "%-25s" % text

