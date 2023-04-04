import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters
import pickle
from skimage.measure import label, regionprops
from os import remove
from skimage.color import label2rgb, rgba2rgb, rgb2gray
from skimage.transform import resize
from PIL import Image, ImageOps
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas





# Usa el clasificador: best_keras_dense


def binarizar(matrix, umbral = 0.25):

    # Binarizar la imagen utilizando la función threshold
    image_binarizada = (matrix > umbral).astype(int).astype(float)
    
    return image_binarizada


def analisis_numero(index, regions_sorted, componentes_conexos, matrix, ver, modelo):
    # toma un index de los componentes conexos, y las regiones de los componentes conexos
        # (el indice esta asociado al componente conexo de un numero)
    # tambien tiene los componentes conexos y la matriz de la imagen
    # obvio tambien tiene el expander de visualizacion
    # tambien tiene el modelo con el que se hacen las predicciones de numero
    # Hace el analisis para predecir cual es este numero
    
    # Añadir a la visualizacion de la imagen, es decir, al expander
    ver.subheader('Analizar componente ' + str(index + 1 ))
    
    # se hace una maskara
    # matriz del mismo tamaño con 1 donde esta el componente conexo del numero
    # y 0 en cualquier otro lado
    mask = np.zeros_like(componentes_conexos)
    mask[componentes_conexos == regions_sorted[index].label] = 1
    mask = mask.astype(float)
    
    # Entonces el numero se consigue:
        #multiplicando esta mask por la matriz de la imagen
        # (de la imagen antes de binarizar)
    imagen_numero = mask * matrix # solo tiene el numero y ceros en otros lados
    
    # Añadir a la visualizacion de la imagen, es decir, al expander
    # ver el numero a analizar en la imagen completa
    col1, col2 = ver.columns(2)
    col1.image(imagen_numero)
    
    
    # Ahora esta imagen se debe de cortar
    # Es decir, meter en una caja
    # donde el numero "choca" en todos los bordes

    # Encontrar los límites del número en la imagen
    rows, cols = np.where(imagen_numero != 0)
    top_row, left_col = np.min(rows), np.min(cols)
    bottom_row, right_col = np.max(rows), np.max(cols)

    # Recortar la imagen utilizando los límites encontrados
    imagen_numero = imagen_numero[top_row:bottom_row+1, left_col:right_col+1]
    
    
    # Ahora se hace cuadrado, rellenando con 0 donde haga falta
    # Calcula la cantidad de ceros que se deben agregar
    n, m = imagen_numero.shape
    pad_width = abs(n-m)//2
    # Agrega ceros a la imagen
    if n > m:
        imagen_numero = np.pad(imagen_numero, ((0,0),(pad_width,pad_width)), mode='constant')
    else:
        imagen_numero = np.pad(imagen_numero, ((pad_width,pad_width),(0,0)), mode='constant')
    
    
    # Ahora esto debe de hacerse de tamaño 20x20
    # Redimensionar la imagen a 20x20 píxeles
    imagen_numero = resize(imagen_numero, (18, 18), anti_aliasing=True)
    
    
    # Ahora se agregar un poco de marco, para que sea al final 28x28
    # donde obvio se agregan filas y columnas llenas de ceros
    imagen_numero = np.pad(imagen_numero, ((5, 5), (5, 5)), 'constant', constant_values=0)
    
    
    # Añadir a la visualizacion de la imagen, es decir, al expander
    # Ver el numero a analizar en una caja 28x28
    # se ve mas claro si se grafica usando una funcion de plt
    # pero este grafico no puede pasarse directamente a streamlit
    # entonces: se crean los graficos usando matplotlib
    #   se salvan como imagenes
    #   esas imageens se suben a streamlit
    #   se eliminan las imagenes de la computadora
    
    plt.axis('off') # quitar ejes
    plt.imshow(imagen_numero, cmap='gray')
    plt.savefig('imagen_numero.jpg', bbox_inches='tight', pad_inches=0.0)
    col2.image('imagen_numero.jpg', use_column_width= True)
    remove("imagen_numero.jpg")
        
        
    # Ahora se predice el numero
    # el modelo best_keras_dense, toma imagenes:
        # que tengan dimensiones de 1x28x28 (pues en general es nx28x28)
        # que tengan rango de 0 a 255
    
    # llevar de (28x28) a (1x28x28)
    imagen_numero = imagen_numero.reshape((1, -28, 28))
    # llevar rango de (0 a 1) de (0 a 255)
    imagen_numero = imagen_numero * 255

    # Predecir    
    prediccion = modelo.predict(imagen_numero)
    # la prediccion es un array de 10 numeros obviamente
    # tomar el numero que esta preiciendo    
    num = np.argmax(prediccion)
    
    # Añadir a la visualizacion de la imagen, es decir, al expander
    ver.write("El numero es: " +str(num))
    
    return num

def analisis_signo(index, regions_sorted, componentes_conexos, bin_matrix, ver):
    # toma un index de los componentes conexos, y las regiones de los componentes conexos
        # (el indice esta asociado al componente conexo de un signo)
    # tambien tiene los componentes conexos y la matriz del signo binaarizada
    # obvio tambien tiene el expander de visualizacion
    # Hace el analisis para predecir cual es este signo

    # Añadir a la visualizacion de la imagen, es decir, al expander
    ver.subheader('Analizar componente ' + str(index + 1 ))
    
    # se hace una maskara
    # matriz del mismo tamaño con 1 donde esta el componente conexo del signo
    # y 0 en cualquier otro lado
    mask = np.zeros_like(componentes_conexos)
    mask[componentes_conexos == regions_sorted[index].label] = 1
    mask = mask.astype(float)
    
    # Entonces el signo se consigue:
        #multiplicando esta mask por la matriz de la imagen
        # (de la imagen antes de binarizar)
    imagen_signo = mask * bin_matrix # solo tiene el signo y ceros en otros lados
    
    # Añadir a la visualizacion de la imagen, es decir, al expander
    # ver el numero a analizar en la imagen completa
    col1, col2 = ver.columns(2)
    col1.image(imagen_signo)
    
    # Encontrar los límites del número en la imagen
    rows, cols = np.where(imagen_signo != 0)
    top_row, left_col = np.min(rows), np.min(cols)
    bottom_row, right_col = np.max(rows), np.max(cols)

    # Recortar la imagen utilizando los límites encontrados
    imagen_signo = imagen_signo[top_row:bottom_row+1, left_col:right_col+1]
    
    len_x = imagen_signo.shape[1] # longuiud de esta imagen "ajustada" en el eje x
    len_y = imagen_signo.shape[0] # longuiud de esta imagen "ajustada" en el eje y
    
    # La proporcion entre len_y y len_x es la que determina el signo
    # Mientras mayor sea len_x con respecto a len_y es mas probable que sea signo -
    
    x_sobre_y = len_x / len_y # mientras mas grande, el signo es mas cercano a un -
    # Notar que obviamente:
        # si es uno es una cruz perfecta
        # menor a uno es mas alto
        # mayor a uno es mas ancho
    
    treshold_signo = 2
    
    if x_sobre_y >= 2:
        signo =  "-"
    else:
        signo =  "+"

    
    # Añadir a la visualizacion de la imagen, es decir, al expander
    # Se visualiza el signo en una "caja" del mismo formato de los numeros
    # todo esto no tiene ningun valor 
    # pues como se puede ver, la prediccion ya se realizao
    # solo es para darle formato y que se vea uniforme con los numeros
    
    # Ver el signo a analizar en una caja 28x28
    # se ve mas claro si se grafica usando una funcion de plt
    # pero este grafico no puede pasarse directamente a streamlit
    # entonces: se crean los graficos usando matplotlib
    #   se salvan como imagenes
    #   esas imageens se suben a streamlit
    #   se eliminan las imagenes de la computadora
    
    # Ahora se hace cuadrado, rellenando con 0 donde haga falta
    # Calcula la cantidad de ceros que se deben agregar
    n, m = imagen_signo.shape
    pad_width = abs(n-m)//2
    # Agrega ceros a la imagen
    if n > m:
        imagen_signo = np.pad(imagen_signo, ((0,0),(pad_width,pad_width)), mode='constant')
    else:
        imagen_signo = np.pad(imagen_signo, ((pad_width,pad_width),(0,0)), mode='constant')
    
    
    # Ahora esto debe de hacerse de tamaño 20x20
    # Redimensionar la imagen a 20x20 píxeles
    imagen_signo = resize(imagen_signo, (20, 20), anti_aliasing=True)
    
    
    # Ahora se agregar un poco de marco, para que sea al final 28x28
    # donde obvio se agregan filas y columnas llenas de ceros
    imagen_signo = np.pad(imagen_signo, ((4, 4), (4, 4)), 'constant', constant_values=0)
    
    
    plt.axis('off') # quitar ejes
    plt.imshow(imagen_signo, cmap='gray')
    plt.savefig('imagen_signo.jpg', bbox_inches='tight', pad_inches=0.0)
    col2.image('imagen_signo.jpg', use_column_width= True)
    remove("imagen_signo.jpg")
    
    ver.write("El signo es: " +str(signo))
        
    return signo
    

def main():
    
    st.title('Proyecto reconocimiento de números')
    
    # Dar la opcion al usuario de cargar una imagen o dibujarla
    # Pedir la selecccion
    st.subheader('Seleccionar como se va a presentar el input')
    option = st.selectbox(
    'label',
    ('Cargar una imagen', 'Dibujar'), label_visibility = "collapsed")
    
    se_puede_analizar = False # pues aun no hay imagen
    # cuando se de una imagen se hara verdadero
    
    # Se va a dibujar la imagen
    if option == "Dibujar":
        st.subheader('Dibuja la operación a calcular')
        st.write('La operacion debe de cumplir que:')
        st.markdown("- Solo contiene numeros del 0 al 9\n"+
                    "- El primero numero es positivo (i.e no se puede -1+2 = )\n"+
                    "- La operacion debe terminar con un signo igual\n"+
                    "- Ademas de los numeros y los signos, no debe haber nada mas en la imagen")
        
        # especificar los parametros para dibujar
        anchura_trazo = st.slider("Anchura del trazo: ", 5, 15, 10)
        color_trazo = st.color_picker("Color del trazo: ")

        # Create a canvas component
        canvas_result = st_canvas(
            stroke_width=anchura_trazo,
            stroke_color=color_trazo,
            background_color="#eee",
            background_image= None,
            update_streamlit= True,
            height=150,
            drawing_mode= "freedraw",
            point_display_radius= 0,
            key="canvas",
            display_toolbar  = True
        )
        
        # photo es tal cual el dibujo del usuario
        # es una imagen rgba
        photo = canvas_result.image_data

        # transformarla a rgb
        photo = rgba2rgb(photo)
        
        # Obtener image_matrix
        # Esta es una matriz en formato de grises (escala de 0 a 1)
        # Es la que se usara para el analisis
        image_matrix  = rgb2gray(photo)
        
        # Solo se puede analizar si no todo es fondo
        # Es decir, si hay algo dibujado
        # 238/255 es el color del fondo
        if (image_matrix != 238/255).any(): # es true si hay algo
            se_puede_analizar = True # ya se puede hacer un analisis del dibujos

        
               
    # Se va a cargar la imagen
    elif option == "Cargar una imagen":
        st.subheader('Cargar la operación a calcular')
        st.write('La operacion debe de cumplir que:')
        st.markdown("- Solo contiene numeros del 0 al 9\n"+
                    "- El primero numero es positivo (i.e no se puede -1+2 = )\n"+
                    "- La operacion debe terminar con un signo igual\n"+
                    "- Ademas de los numeros y los signos, no debe haber nada mas en la imagen\n"
                    "- El fondo debe de ser blanco")

        # dejar al usuario subir una foto
        photo = st.file_uploader('', type  =  ['png', 'jpg'])
        
        # photo es tal cual la imagen que suve el usuario
        if photo is not None:
            st.image(photo) # ver la imagen que se subio
            
            # Obtener image_matrix
            # Esta es una matriz en formato de grises (escala de 0 a 1)
            # Es la que se usara para el analisis
            image_matrix = io.imread(photo, as_gray = True)
            
            se_puede_analizar = True # ya se puede hacer un analisis de la imagen
            
            

    # para este punto, ya sea si se subio o dibujo la imagen
    # ya se tiene la variable de photo (tal cual como se subio/dibujo)
    # y la variable image_matrix (escala de grises) que se usara para analisis
    
    if se_puede_analizar:
        if st.button('Analizar operacion'):
            
            # para que el resultado sea lo primero que se ve
            espacio_resultado = st.empty()
            
            # obtener una matriz, de 0 a 1, formato de grises
            #image_matrix = io.imread(photo, as_gray = True)
            
            # Se asume que el fondo es blanco, debe de ser negro
            matrix = 1 - image_matrix # hacer el fondo negro, letra blanca
            
            # Las visualizaciones del analisis se ponene en un expander
            # Asi el usuario puede decidir si verlas o no
            # Crear el expander
            ver = st.expander('Ver el Analisis de la Imagen')
            # Añadir a la visualizacion de la imagen, es decir, al expander
            ver.header('Desglose del analisis de la imagen:')
            ver.subheader('Imagen con fondo negro, letra blanca')
            ver.image(matrix)
                
            # Aumentar contraste
            # utiliza los percentiles 2 y 98 de la imagen original como límites para el rango de valores de la imagen resultante.
            p2, p98 = np.percentile(matrix, (2, 98))
            matrix = exposure.rescale_intensity(matrix, in_range=(p2, p98))
            
            # Añadir a la visualizacion de la imagen, es decir, al expander
            ver.subheader('Aumentar constraste')
            ver.image(matrix)
    
    
            # Binarizar la imagen
            bin_matrix = binarizar(matrix)
            
            # Etiquetar los componentes conexos
            componentes_conexos, cantidad = label(bin_matrix, background = 0, return_num=True)
            # Componentes conexos, es la misma imagen pero 
            # en lugar de 0 y 1, tiene distintos numeros enteros
            # dependiendo del numero de componente que sea
            
            # Añadir a la visualizacion de la imagen, es decir, al expander
            ver.subheader('Identificar los componentes conexos')
            # Colorear los diferentes compoenntes conexos
            # especificar que los que no perteneces a ninguno (tienen numero 0)
            # sean coloreados de negro
            colored_image = label2rgb(componentes_conexos, bg_label=0)
            ver.write('Numero de componentes: ' + str(cantidad))
            ver.image(colored_image)
                
            
            
            # Aca puede haber componentes conexos que sean muy pequeños
            # Manchas o asi, solo considerar componentes conexos
            # tales que sean mayores a cierto porcentaje de la imagen
            
            min_porcentaje = 0.02 # los que sean menor, se eliminan
            min_tamaño = min_porcentaje/100 * (bin_matrix.shape[0]*bin_matrix.shape[1])
            
            # Obtener las propiedades de los componentes conexos
            props = regionprops(componentes_conexos)
                    
            # Eliminar los componentes conexos de tamaño pequeño
            for prop in props:
                if prop.area < min_tamaño: # si se tiene menor area
                    # hacer que no pertenezcar a ningun componente conexo
                    componentes_conexos[componentes_conexos == prop.label] = 0
                    
            # Voler a encontrar los componentes conexos (ya con los pequeños eliminados)
            componentes_conexos, cantidad = label(componentes_conexos, background = 0, return_num=True)
            # Ahora si se tienen solo los componentes conexos "importantes"
            
            # Añadir a la visualizacion de la imagen, es decir, al expander
            ver.subheader('Conservar solo los componentes conexos grandes')
            # Colorear los diferentes compoenntes conexos
            # especificar que los que no perteneces a ninguno (tienen numero 0)
            # sean coloreados de negro
            colored_image = label2rgb(componentes_conexos, bg_label=0)
            ver.write('Numero de componentes: ' + str(cantidad))
            ver.image(colored_image)
                                
            # Obtener las coordenadas de los píxeles de cada componente conexo
            regions = regionprops(componentes_conexos)
            
            # Hacer que el signo igual ya no sea considerado como componente conexo
            # Asi qeu hay que quitar dos componentes, uno por cada linea
            
            # Ordenar los componentes conexos por su coordenada x más grande 
            regions_sorted = sorted(regions, key=lambda r: r.bbox[1])
            
            # Eliminar los dos mas a la derecha
            componentes_conexos[componentes_conexos == regions_sorted[-1].label] = 0
            componentes_conexos[componentes_conexos == regions_sorted[-2].label] = 0
            
            cantidad = cantidad - 2 # pues ya no se considera el signo igual
            assert cantidad % 2 == 1 # ver que sea impar, tiene que serlo
            cant_num = (cantidad + 1) // 2 # obvio es entero, cantidad de numeros
            cant_signo = (cantidad - 1) // 2 # obvio es entero, cantidad de signos
            
            
            # Añadir a la visualizacion de la imagen, es decir, al expander
            ver.subheader('No contemplar el signo de igual')
            # Colorear los diferentes compoenntes conexos
            # especificar que los que no perteneces a ninguno (tienen numero 0)
            # sean coloreados de negro
            colored_image = label2rgb(componentes_conexos, bg_label=0)
            ver.write('Numero de componentes: ' + str(cantidad))
            ver.write("Cantidad de numeros: " + str(cant_num))
            ver.write("Cantidad de signos: " + str(cant_signo))
            ver.image(colored_image)
            
            ver.subheader("Analisis componente a componente")
            
    
            # Obviamente la operacion se puede dividir en:
                # Primero un numero (x0)
                # Despues una cantidad k de operaciones (+- xi) de igual a i hasta k
                # Por ejemplo, si se tiene 6-5+1 se divide en: (6) (-5) (+1)
                # Donde lo que va en cada parentesis se analiza por separado
                
            # EL resultado se va actualizando cada que se analiza un par nuevo
            # Variables donde se va a ir actualizando
            result_str = "" # resultado en cadena de texto, por ejemplo: "6-5+1"
            result_num = 0  # resutlado numerico, por ejemplo: 2
                
            # Las predicciones de los numeros se hacen con este modelo
            modelo = load_model('best_keras_dense.h5')
    
        
            # Primero se ve el primer numero
            predict_x0 = analisis_numero(0, regions_sorted, componentes_conexos, bin_matrix, ver, modelo)
            # Actualizar las variables de los resultados
            result_str = result_str + str(predict_x0)
            result_num = result_num + predict_x0
                    
            
    
            # Se empiezan a analizar los k pares de signo-operacion
            k = cant_signo # obvio se analizan tantos pares como operaciones
            # hacer un for que va [1, 2, ..., k]
            for i in range(1, k+1):
                # Analisar el signo de este par, que es el componente conexo (2*i-1)
                
                predict_signo_i = analisis_signo(2*i-1, regions_sorted, componentes_conexos, bin_matrix, ver)
                
                
                
                # Analisar el numero de este par, que es el componente conexo (2*i)
                
                # Predecir cual es este numero
                predict_xi = analisis_numero(2*i, regions_sorted, componentes_conexos, matrix, ver, modelo)
                
                
                # Actualizar las variables de los resultados
                if predict_signo_i == "+":
                    result_num = result_num + predict_xi
                    result_str = result_str + " + " + str(predict_xi)
                elif predict_signo_i == "-":
                    result_num = result_num - predict_xi
                    result_str = result_str + " - " + str(predict_xi)
    
    
    
            # Dar el resultado en le expander
            ver.subheader("Por lo tanto, el resultado es: " + result_str + " = " + str(result_num))
            # Tambien ponerlo antes del analisis
            espacio_resultado.header("Resultado: " + result_str + " = " + str(result_num))


if __name__ == '__main__':
    main()

        













