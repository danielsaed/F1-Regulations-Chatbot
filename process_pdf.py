import pdfplumber
import re
import fitz
import json

def process_international_sporting_code():

    import json
    import pdfplumber
    import re

    texto_completo = ""

    with pdfplumber.open(r"pdf/2025_international_sporting_code_fr-en_0.pdf") as pdf:

        for idx,page in enumerate(pdf.pages):
            if idx <1:
                continue
            # Define las coordenadas de las columnas
            ancho_pagina = page.width
            mitad_pagina = ancho_pagina / 2 - 10
            # Define la caja de la columna derecha (francés)
            columna_derecha = (mitad_pagina, 0, ancho_pagina, page.height)

            # Extrae el texto de cada columna
            texto_completo += page.within_bbox(columna_derecha).extract_text()

    texto_completo = re.sub(r'(\w)-\n(\w)', r'\1\2', texto_completo) # Une palabras cortadas con guion
    texto_completo = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto_completo) # Une saltos de línea simples

    texto_completo = re.sub(r'Application from 1st January 2025', '', texto_completo)

    section_titles = ['ARTICLE 1 GENERAL', 'ARTICLE 2 COMPETITIONS', 'ARTICLE 3 COMPETITIONS', 'ARTICLE 4 TOURING', 'ARTICLE 5 PARADE', 'ARTICLE 6', 'ARTICLE 7', 'ARTICLE 8', 'ARTICLE 9', 'ARTICLE 10', 'ARTICLE 11', 'ARTICLE 12', 'ARTICLE 13', 'ARTICLE 14', 'ARTICLE 15', 'ARTICLE 16', 'ARTICLE 17', 'ARTICLE 18', 'ARTICLE 19']
    articles_text = [] # Este será tu array final con los 64 textos


    # Iteramos sobre la lista de títulos con un índice
    for i in range(len(section_titles)):

        current_title = section_titles[i]

        # Encontramos la posición donde empieza el título actual
        start_pos = texto_completo.find(current_title)

        # Si el título no se encuentra, lo saltamos y avisamos.
        if start_pos == -1:
            print(f"ADVERTENCIA: No se encontró el título '{current_title}' en el texto.")
            continue

        # Determinamos dónde termina el texto de este artículo.
        # Por defecto, es el final del documento.
        end_pos = len(texto_completo)

        # Si no es el último título de la lista, el final es donde empieza el SIGUIENTE título.
        if i + 1 < len(section_titles):
            next_title = section_titles[i+1]
            next_title_pos = texto_completo.find(next_title)
            if next_title_pos != -1:
                end_pos = next_title_pos

        # "Cortamos" el texto del artículo completo, desde su título hasta el inicio del siguiente.
        article_content = texto_completo[start_pos:end_pos]

        # Añadimos el texto limpio (sin espacios extra al inicio/final) al array
        articles_text.append(article_content.strip())

    lst_modify_art = []
    lst_regulation_names = []
    for i in articles_text:
        indice=i[1:].find("ARTICLE")
        lst_modify_art.append(i[indice:])
        lst_regulation_names.append(i[10:indice].strip())

    counter = 0
    dic = {}
    for idx,i in enumerate(lst_modify_art):
        counter = 1
        dic[idx+1] = []
        while True:
            indice_start = i.find("ARTICLE "+str(idx+1)+"."+str(counter))
            indice_end = i.find("ARTICLE "+str(idx+1)+"."+str(counter+1))

            if indice_start == -1:
                break
            counter=counter+1
            if indice_end == -1:
                dic[idx+1].append(i[indice_start:])
            else:
                dic[idx+1].append(i[indice_start:indice_end])
            i = i[indice_end:]

    lst_db = []

    for i in dic:
        temp_dic = {
            "article_number":i,
            "regulation_name":lst_regulation_names[i-1],
            "subarticles":[]
        }
    for idx,j in enumerate(dic[i]):
        temp_sub_dic = {
            "subarticle_number":str(i)+"."+str(idx+1),
            "document":"INTERNATIONAL SPORTING CODE",
            "subarticle_text":j
        }

        temp_dic["subarticles"].append(temp_sub_dic)

    lst_db.append(temp_dic)

    dic_database = {}

    for i in lst_db:
        #dic_database["INTERNATIONAL SPORTING CODE"+"-"+i["regulation_name"]] = f"Document: INTERNATIONAL SPORTING CODE, Article #{i["article_number"]}, Regulation: {i["regulation_name"]} /n {"/n".join([sub["subarticle_text"] for sub in i["subarticles"]])}"
        dic_database["INTERNATIONAL SPORTING CODE"+"-"+i["regulation_name"]] = f"Document: INTERNATIONAL SPORTING CODE, Article #{i['article_number']}, Regulation: {i['regulation_name']} /n {'/n'.join([sub['subarticle_text'] for sub in i['subarticles']])}"

    file_path = "INTERNATIONAL SPORTING CODE.json"

    # --- Aquí ocurre la magia ---
    # Abrimos el archivo en modo de escritura ('w' de write)
    # encoding='utf-8' es crucial para manejar caracteres especiales como acentos o símbolos
    with open(file_path, 'w', encoding='utf-8') as json_file:
        # Usamos json.dump() para escribir la lista en el archivo
        # indent=4 -> Hace que el archivo sea legible para humanos, con 4 espacios de sangría
        # ensure_ascii=False -> Permite que caracteres no-ASCII (como ñ, á, é) se guarden como son
        json.dump(lst_db, json_file, indent=4, ensure_ascii=False)

def process_f1_sporting_regulations():
    # Ejemplo conceptual con PyMuPDF
    import fitz  # PyMuPDF
    import re
    import json

    doc = fitz.open(r"pdf/FIA 2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30.pdf")
    full_text = ""
    for idx,page in enumerate(doc):
        if idx <1 or idx > 73:
            continue
        full_text += page.get_text()


    # Eliminar el footer de cada página (ejemplo)
    cleaned_text = re.sub(r'©\d{4} Fédération Internationale de l’Automobile\s+Issue \d', '', full_text)
    # Eliminar el header (hay que ajustar el patrón exacto)
    cleaned_text = re.sub(r'2025 Formula 1 Sporting Regulations\s+\d+/\d+\s+\d{2} April \d{4}', '', cleaned_text)
    # Unir líneas que fueron cortadas
    cleaned_text = re.sub(r'(\w)-\n(\w)', r'\1\2', cleaned_text) # Une palabras cortadas con guion
    cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text) # Une saltos de línea simples

    title_pattern = r'\d+\)\s+[A-Z]{2,}'
    # Encontramos todas las ocurrencias de los títulos
    titles = re.findall(title_pattern, cleaned_text)

    section_titles = ['1)  REGULATIONS', '2)  GENERAL', '3)  GENERAL', '4)  LICENCES', '5)  CHAMPIONSHIP', '6)  WORLD', '7)  DEAD', '8)  COMPETITORS', '9)  CAR', '10)  TRACK', '11)  PROMOTER', '12)  ORGANISATION', '13)  INSURANCE', '14)  FIA', '15)  OFFICIALS', '16)  INSTRUCTIONS', '17)  PROTESTS', '18)  SANCTIONS', '19)  PRESS', '20)  MEETINGS', '21)  COVERING', '22)  DRIVER', '23)  OPERATIONAL', '24)  COMPETITOR', '25)  POWER', '26)  GENERAL', '27)  SPARE', '28)  POWER', '29)  VOID', '30)  SUPPLY', '31)  SCRUTINEERING', '32)  CHANGES', '33)  DRIVING', '34)  PIT', '35)  WEIGHING', '36)  REFUELLING', '37)  PRACTICE', '38)  FREE', '39)  QUALIFYING', '40)  PRE', '41)  VOID', '42)  THE', '43)  SPRINT', '44)  RACE', '45)  EXTRA', '46)  DELAYED', '47)  ABORTED', '48)  FALSE', '49)  FORMATION', '50)  STARTING', '51)  STANDING', '52)  ROLLING', '53)  SPRINT', '54)  INCIDENTS', '55) SAFETY', '56)  VIRTUAL', '57)  SUSPENDING', '58)  RESUMING', '59)  FINISH', '60)  POST', '61)  SPRINT', '62)  RACE', '63)  PODIUM', '64)  TEAM']
    articles_text = [] # Este será tu array final con los 64 textos

    # Iteramos sobre la lista de títulos con un índice
    for i in range(len(section_titles)):

        current_title = section_titles[i]

        # Encontramos la posición donde empieza el título actual
        start_pos = cleaned_text.find(current_title)

        # Si el título no se encuentra, lo saltamos y avisamos.
        if start_pos == -1:
            print(f"ADVERTENCIA: No se encontró el título '{current_title}' en el texto.")
            continue

        # Determinamos dónde termina el texto de este artículo.
        # Por defecto, es el final del documento.
        end_pos = len(cleaned_text)

        # Si no es el último título de la lista, el final es donde empieza el SIGUIENTE título.
        if i + 1 < len(section_titles):
            next_title = section_titles[i+1]
            next_title_pos = cleaned_text.find(next_title)
            if next_title_pos != -1:
                end_pos = next_title_pos

        # "Cortamos" el texto del artículo completo, desde su título hasta el inicio del siguiente.
        article_content = cleaned_text[start_pos:end_pos]

        # Añadimos el texto limpio (sin espacios extra al inicio/final) al array
        articles_text.append(article_content.strip())


    lst_modify_art = []
    lst_regulation_names = []
    for i in articles_text:
        indice=i[2:].find(".")-1
        lst_modify_art.append(i[indice:])
        lst_regulation_names.append(i[2:indice].replace(")","").strip())

    counter = 0
    dic = {}
    for idx,i in enumerate(lst_modify_art):
        counter = 1
        dic[idx+1] = []
        while True:
            indice_start = i.find(str(idx+1)+"."+str(counter))
            indice_end = i.find(str(idx+1)+"."+str(counter+1))

            if indice_start == -1:
                break
            counter=counter+1
            if indice_end == -1:
                dic[idx+1].append(i[indice_start:])
            else:
                dic[idx+1].append(i[indice_start:indice_end])
            i = i[indice_end:]

    lst_db = []

    for i in dic:
        temp_dic = {
            "article_number":i,
            "regulation_name":lst_regulation_names[i-1],
            "subarticles":[]
        }
    for idx,j in enumerate(dic[i]):
        temp_sub_dic = {
            "subarticle_number":str(i)+"."+str(idx+1),
            "document":"FORMULA ONE SPORTING REGULATIONS",
            "subarticle_text":j
        }

        temp_dic["subarticles"].append(temp_sub_dic)

    lst_db.append(temp_dic)

    dic_database={}
    for i in lst_db:
        dic_database["FORMULA ONE SPORTING REGULATIONS"+"-"+i["regulation_name"]] = f"Document: FORMULA ONE SPORTING REGULATIONS, Article #{i['article_number']}, Regulation: {i['regulation_name']} /n {'/n'.join([sub['subarticle_text'] for sub in i['subarticles']])}"

    file_path = "FORMULA ONE SPORTING REGULATIONS.json"

    # --- Aquí ocurre la magia ---
    # Abrimos el archivo en modo de escritura ('w' de write)
    # encoding='utf-8' es crucial para manejar caracteres especiales como acentos o símbolos
    with open(file_path, 'w', encoding='utf-8') as json_file:
        # Usamos json.dump() para escribir la lista en el archivo
        # indent=4 -> Hace que el archivo sea legible para humanos, con 4 espacios de sangría
        # ensure_ascii=False -> Permite que caracteres no-ASCII (como ñ, á, é) se guarden como son
        json.dump(lst_db, json_file, indent=4, ensure_ascii=False)

def process_international_sporting_code1():

    import json
    import pdfplumber
    import re



    section_titles = ['ARTICLE 1 GENERAL', 'ARTICLE 2 COMPETITIONS', 'ARTICLE 3 COMPETITIONS', 'ARTICLE 4 TOURING', 'ARTICLE 5 PARADE', 'ARTICLE 6', 'ARTICLE 7', 'ARTICLE 8', 'ARTICLE 9', 'ARTICLE 10', 'ARTICLE 11', 'ARTICLE 12', 'ARTICLE 13', 'ARTICLE 14', 'ARTICLE 15', 'ARTICLE 16', 'ARTICLE 17', 'ARTICLE 18', 'ARTICLE 19']
    # Este será tu array final con los 64 textos

    #C1 2-33
    #C2 34-46
    #C3 47-54
    #C4 55-58
    #C5 59-60

    dic_titles = {
        #"APPENDIX L CHAPTER I - FIA INTERNATIONAL DRIVERS’ LICENCES":[[1,32],["1. General","2. Type","3. International","4. International","5. International","6. International","7. International","8. International","9. International","10. International","11. International","12. International","13. Qualification","14.  Qualification","15. Licences","18.  Licences"]],
        #"APPENDIX L CHAPTER II - REGULATIONS FOR THE MEDICAL EXAMINATION OF DRIVERS":[[33,45],["1. Annual","2. Medical","4. Appeals","5. Regulations"]],
        "APPENDIX L CHAPTER III DRIVERS EQUIPMENT":[[46,53],["1. Helmets","2. Flame","3. Frontal","4. Safety","5. Wearing"]],
        "APPENDIX L CHAPTER IV CODE OF DRIVING CONDUCT ON CIRCUITS":[[54,57],["2. Ove","3. Car","4. Entrance","5. Pit","6. Exit"]],
    }
    

    # Iteramos sobre la lista de títulos con un índice
    for key in dic_titles:
        articles_text = [] 
        texto_completo = ""

        with pdfplumber.open(r"pdf\appendix_l_2025_publie_le_10_juin_2025.pdf") as pdf:

            for idx,page in enumerate(pdf.pages):
                
                if idx < dic_titles[key][0][0] or idx > dic_titles[key][0][1]:
                    print(idx)
                    continue

                # Define las coordenadas de las columnas
                
                ancho_pagina = page.width
                mitad_pagina = ancho_pagina / 2 - 7
                # Define la caja de la columna derecha (francés)
                columna_derecha = (mitad_pagina, 0, ancho_pagina, page.height)

                # Extrae el texto de cada columna
                texto_completo += page.within_bbox(columna_derecha).extract_text()

        texto_completo = re.sub(r'(\w)-\n(\w)', r'\1\2', texto_completo) # Une palabras cortadas con guion
        texto_completo = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto_completo) # Une saltos de línea simples

        texto_completo = re.sub(r'ANNEXE L  APPENDIX L ', '', texto_completo)

        print(texto_completo)

            
        section_titles = dic_titles[key][1]

        for i in range(len(section_titles)):

            current_title = section_titles[i]

            # Encontramos la posición donde empieza el título actual
            start_pos = texto_completo.find(current_title)

            # Si el título no se encuentra, lo saltamos y avisamos.
            if start_pos == -1:
                print(f"ADVERTENCIA: No se encontró el título '{current_title}' en el texto.")
                continue

            # Determinamos dónde termina el texto de este artículo.
            # Por defecto, es el final del documento.
            end_pos = len(texto_completo)

            # Si no es el último título de la lista, el final es donde empieza el SIGUIENTE título.
            if i + 1 < len(section_titles):
                next_title = section_titles[i+1]
                next_title_pos = texto_completo.find(next_title)
                if next_title_pos != -1:
                    end_pos = next_title_pos

            # "Cortamos" el texto del artículo completo, desde su título hasta el inicio del siguiente.
            article_content = texto_completo[start_pos:end_pos]

            # Añadimos el texto limpio (sin espacios extra al inicio/final) al array
            articles_text.append(article_content.strip())
        
        
        lst_modify_art = []
        lst_regulation_names = []
        lst_art_number = []
        dic_regulation_name={}

        for i in articles_text:
            indice=i[2:60].find(".")
            print(indice)
            if indice == -1:
                indice=i[2:60].find("a)")
            lst_modify_art.append(i[indice+1:])

            #lst_regulation_names.append(i[3:indice].strip())
            lst_art_number.append(i[:i.find(".")])

            dic_regulation_name[lst_art_number[-1]] = i[3:indice+1].strip()
        print(dic_regulation_name)
        print(lst_art_number)
        

        counter = 0
        dic = {}
        for idx,i in enumerate(lst_modify_art):
            counter = 1
            dic[lst_art_number[idx]] = []
            
            while True:
                indice_start = i.find(str(lst_art_number[idx])+"."+str(counter))
                indice_end = i.find(str(lst_art_number[idx])+"."+str(counter+1))

                if indice_start == -1:
                    print("test")
                    if i.find("a)"):
                        dic[lst_art_number[idx]].append(i)
                    break

                counter=counter+1
                if indice_end == -1:
                    dic[lst_art_number[idx]].append(i[indice_start:])
                else:
                    dic[lst_art_number[idx]].append(i[indice_start:indice_end])
                i = i[indice_end:]

        lst_db = []


        for i in dic:
            
            temp_dic = {
                "article_number":i,
                "regulation_name":dic_regulation_name[str(int(i))],
                "subarticles":[]
        }
            
            for idx,j in enumerate(dic[i]):
                temp_sub_dic = {
                    "subarticle_number":str(i)+"."+str(idx+1),
                    "document":key,
                    "subarticle_text":j
                }

                temp_dic["subarticles"].append(temp_sub_dic)

            lst_db.append(temp_dic)

        dic_database = {}

        for i in lst_db:
            dic_database[key+"-"+i["regulation_name"]]  = f"Document: {key}, Article #{i['article_number']}, Regulation: {i['regulation_name']} /n {'/n'.join([sub['subarticle_text'] for sub in i['subarticles']])}"

        file_path = f"{key}.json"
        print(lst_db)
        print(len(lst_db))


        print("-----------------")


        # --- Aquí ocurre la magia ---
        # Abrimos el archivo en modo de escritura ('w' de write)
        # encoding='utf-8' es crucial para manejar caracteres especiales como acentos o símbolos
        with open(file_path, 'w', encoding='utf-8') as json_file:
            # Usamos json.dump() para escribir la lista en el archivo
            # indent=4 -> Hace que el archivo sea legible para humanos, con 4 espacios de sangría
            # ensure_ascii=False -> Permite que caracteres no-ASCII (como ñ, á, é) se guarden como son
            json.dump(lst_db, json_file, indent=4, ensure_ascii=False)

process_international_sporting_code1()