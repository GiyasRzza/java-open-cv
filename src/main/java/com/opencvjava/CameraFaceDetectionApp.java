package com.opencvjava;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class CameraFaceDetectionApp extends Application {

    static {
        System.load("E:\\java-open-cv\\src\\main\\resources\\libs\\opencv_java460.dll");
    }

    private CascadeClassifier faceCascade;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        faceCascade = new CascadeClassifier("E:\\java-open-cv\\src\\main\\resources\\libs\\haarcascade_frontalcatface_extended.xml");

        VideoCapture capture = new VideoCapture(0);

        if (!capture.isOpened()) {
            System.out.println("Could not open camera");
            return;
        }

        Mat frame = new Mat();
        Canvas canvas = new Canvas(640, 480);
        GraphicsContext gc = canvas.getGraphicsContext2D();

        Thread cameraThread = getCameraThread(capture, frame, gc);
        cameraThread.start();

        StackPane root = new StackPane();
        root.getChildren().add(canvas);

        Scene scene = new Scene(root, 640, 480);
        primaryStage.setTitle("Java OpenCV");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private Thread getCameraThread(VideoCapture capture, Mat frame, GraphicsContext gc) {
        Thread cameraThread = new Thread(() -> {
            while (true) {
                capture.read(frame);
                if (!frame.empty()) {
                    detectFaceAndDisplay(frame, gc);

                    Image imageToDisplay = matToImage(frame);

                    Platform.runLater(() -> {
                        if (imageToDisplay != null) {
                            gc.clearRect(0, 0, 640, 480);
                            gc.drawImage(imageToDisplay, 0, 0);
                        }
                    });
                } else {
                    System.out.println("Empty frame!");
                }
            }
        });
        cameraThread.setDaemon(true);
        return cameraThread;
    }

    private Image matToImage(Mat mat) {
        try {
            BufferedImage bufferedImage = matToBufferedImage(mat);
            return convertToFXImage(bufferedImage);
        } catch (IOException e) {
            System.out.println("Image err: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private BufferedImage matToBufferedImage(Mat mat) throws IOException {
        int width = mat.width();
        int height = mat.height();
        int channels = mat.channels();
        byte[] data = new byte[width * height * channels];

        mat.get(0, 0, data);

        BufferedImage image;
        if (mat.channels() == 1) {
            image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        } else { // RGB
            image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        }

        image.getRaster().setDataElements(0, 0, width, height, data);
        return image;
    }

    private Image convertToFXImage(BufferedImage bufferedImage) {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(bufferedImage, "png", baos);
            baos.flush();
            byte[] imageData = baos.toByteArray();
            baos.close();

            return new Image(new ByteArrayInputStream(imageData));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private void detectFaceAndDisplay(Mat frame, GraphicsContext gc) {
        Mat grayImage = new Mat();
        Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

        Imgproc.equalizeHist(grayImage, grayImage);

        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, new org.opencv.core.Size(30, 30), new org.opencv.core.Size());

        Rect[] facesArray = faces.toArray();
        if (facesArray.length == 0) {
            System.out.println("Face not found");
        } else {

            System.out.println("Face found!!!: " + facesArray.length);
        }

        Platform.runLater(() -> {
            for (Rect rect : facesArray) {
                gc.setStroke(javafx.scene.paint.Color.GREEN);
                gc.setLineWidth(3);
                gc.strokeRect(rect.x, rect.y, rect.width, rect.height);

                gc.setFill(javafx.scene.paint.Color.RED);
                gc.setFont(javafx.scene.text.Font.font(20));
                gc.fillText("Face found!!!", rect.x + 5, rect.y - 10);
            }
        });

    }
}