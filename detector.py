import cv2
import torch

# cargar el modelo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# hacer que runee el video en frames para predecir en cada uno
def detector():
    cap = cv2.VideoCapture('Data/people.mp4')
    
    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        # Inferencia de frame
        pred = model(frame)
        # xmin, ymin, xmax, ymax (coordenadas de las cajitas) 
        # predicciones transformadas a pandas, boundin boxes
        df = pred.pandas().xyxy[0]
        # Filtrar por confidence, dejar solo las predicciones mayores a 0.5
        df = df[df['confidence'] > 0.5]

        # para que se vean las cajitas en el video
        # primero transforma los valores a enteros
        for i in range(df.shape[0]):
            bbox = df.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)

            # print bboxes: frame -> (xmin, y min, xmax, ymax) pintandole el bbox a cada frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (225, 0, 0), 2)
            # print text, este caso le printeamos name que es la clase y la confidence que es la prob
            cv2.putText(frame,
                        f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'],4)}",
                        (bbox[0], bbox[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN, # tipo de fuente
                        1, 
                        (255, 255, 255), #color de la letra
                        2) # tamaño de la letra
    

        cv2.imshow("frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()

if __name__ == '__main__':
    detector()

# frame es lo que se va a meter al modelo para que trabaje
