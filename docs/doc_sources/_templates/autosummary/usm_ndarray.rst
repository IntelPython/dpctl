{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}



.. autoclass:: {{ name }}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: generated
   {% for item in methods if item != "__init__" %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree: generated
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
      ~{{name}}.__dlpack_device__
      ~{{name}}.__dlpack__
      ~{{name}}.__sycl_usm_array_interface__
   {% endif %}
   {% endblock %}
