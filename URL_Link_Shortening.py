!pip install pyshorteners

import ipywidgets as widgets
from IPython.display import display
import pyshorteners

def shorten_url(btn):
    original_url = url_input.value.strip()
    if original_url:
        try:
            shortener = pyshorteners.Shortener()
            shortened_url = shortener.tinyurl.short(original_url)
            result_output.value = shortened_url
        except Exception as e:
            result_output.value = f"Error: {str(e)}"
    else:
        result_output.value = "Please enter a valid URL"

def clear_output(btn):
    url_input.value = ""
    result_output.value = ""

# Text input for URL
url_input = widgets.Text(placeholder='Enter URL', description='URL:', layout=widgets.Layout(width='80%'))

# Button to shorten URL
shorten_button = widgets.Button(description='Shorten', layout=widgets.Layout(width='20%'))
shorten_button.on_click(shorten_url)

# Output area for displaying result
result_output = widgets.Textarea(description='Shortened URL:', layout=widgets.Layout(width='100%', height='100px'), disabled=True)

# Button to clear input and output
clear_button = widgets.Button(description='Clear', layout=widgets.Layout(width='100%'))
clear_button.on_click(clear_output)

# Display UI elements
display(url_input)
display(widgets.HBox([shorten_button, clear_button]))
display(result_output)
