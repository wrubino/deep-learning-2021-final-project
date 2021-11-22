from IPython.core.display import display, Markdown


def get_obj_attributes(obj):
    """
    A function that returns a dict of object attributes.
    :param obj:
    :type obj:
    :return:
    :rtype:
    """
    return {attribute_name: getattr(obj, attribute_name)
            for attribute_name in dir(obj)
            if (not attribute_name.startswith('__')
                and not callable(getattr(obj, attribute_name)))}


def get_obj_methods(obj):
    """
    A function that returns a dict of object methods.
    :param obj:
    :type obj:
    :return:
    :rtype:
    """
    return {method_name: getattr(obj, method_name)
            for method_name in dir(obj)
            if (not method_name.startswith('__')
                and callable(getattr(obj, method_name)))}


def printmd(markdown_text,
            font_family='courier',
            font_size=14):
    """
        A function that prints a string in markdown format.
        :param string:
        :type string:
        :return:
        :rtype:
        """
    # Initialize the html formatting markup.
    pre_html = '<span style="'

    # Set the font family in the HTML markup.
    if font_family is not None:
        pre_html += f'font-family: {font_family.lower()}; '
        html_applied = True

    # Set the font size in the HTML markup.
    if font_size is not None:
        pre_html += f'font-size: {font_size}px; '
        html_applied = True

    # Finish the HTML markup
    pre_html += '">'
    post_html = '</span>'

    # Create the final text string to be displayed.
    final_markdown_text = f'{pre_html}{markdown_text}{post_html}'

    # Display.
    display(Markdown(final_markdown_text))


def unique(list_):
    """
    A function that returns unique values from a list.
    :param list_:
    :type list_:
    :return:
    :rtype:
    """
    return list(set(list_))
