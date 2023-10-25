from typing import List, Optional, Union, Dict, Any

from haystack.preview import component, Document, default_to_dict


@component
class GradientDocumentEmbedder:
    """
    A component for computing Document embeddings using Gradient AI API..
    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    pass

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        return {"documents": []}
