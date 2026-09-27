################################################################
# Custom extension to automatically document configuration options by processing the
# data in blip/config.py.
################################################################

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
import blip.config

# TODO make this generate the prettier `confval` sphinx directives.
# TODO also document spectral and spatial models.
# This tutorial may be of help:
# https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html

class BlipConfigDirective(SphinxDirective):
    """
    Usage:
        .. blip-config-section:: SECTION_PARAMS
    
    Get the list of parameters and render it in RST.
    """

    required_arguments = 1

    def run(self):
        opts: list[blip.config.Option] = getattr(blip.config, self.arguments[0])
        # show required options first
        opts_required = [opt for opt in opts if opt.required]
        opts_not_requ = [opt for opt in opts if not opt.required]
        opts = opts_required + opts_not_requ

        optnodes = []
        for opt in opts:
            t1 = f".. confval:: {opt.name}\n"
            t2 = f"    :default: {opt.default}\n" if not opt.required else ""
            t3 = f"\n    {opt.desc}\n"
            text = t1+t2+t3
            parsed = self.parse_text_to_nodes(text)
            optnodes.extend(parsed)
        return optnodes


def setup(app: Sphinx):
    app.add_directive("blip-config-section", BlipConfigDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
