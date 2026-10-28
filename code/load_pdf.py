from pymupdf import *
from pathlib import Path   
import numpy as np   
import ollama                          

def clean_text(text: str) -> str:
    """Nettoyage léger : enlève les espaces en trop et les lignes vides multiples."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # supprime les lignes vides
    return "\n".join(lines)

def load_pdf(file_path) : #Charge le pdf et retourne les pages
    """Charge les pdf dans autant de tuple qu'il y a de page (nom_doc, num_page, texte_page)"""
    with pymupdf.open(file_path) as document : 
        liste_page = []
        nom = Path(file_path).name
        for page in document : 
            liste_page.append([nom, page.number + 1, clean_text(page.get_text())])
    return liste_page

def chunking(list_page, size_chunk, depassement) :        #Decoupe les pages en chunk 
    #size_chunk = 600                                     #taille des morceaux
    #depassement = 150                                   #Overlap
    list_chuck = []
    for page in list_page : 
        longueur = len(page[2])
        i = 0
        while i < longueur : 
            list_chuck.append([page[0], page[1], page[2][(i + size_chunk) : min(i + size_chunk, longueur - 1)]])
            i += size_chunk - depassement
    return list_chuck

def embed(list_chuck) : #Prend la list_chuck et vectorise le texte
    list_texte = []
    info_docpage = []
    for elt in list_chuck : 
        list_texte.append(elt[2])
        info_docpage.append([elt[0], elt[1]])
    vectored = ollama.embed(model="bge-m3", input = list_texte)
    vectorisation = np.array(vectored.embeddings)
    return (vectorisation, info_docpage)

print(embed(chunking(load_pdf("Doc_info/guide_asso_test.pdf"), 600, 150)))

        
    
    





    