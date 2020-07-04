package com.example.trafficsigndetection;

import android.annotation.TargetApi;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.Trace;
import android.util.DisplayMetrics;

import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.annotation.Annotation;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
//import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.gpu.GpuDelegate;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
    //private static final Logger LOGGER = new Logger();

    // Only return this many results.
    private static final int NUM_DETECTIONS = 10;
    // Float model
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    // Number of threads in the java app
    private static int NUM_THREADS = 1;
    private boolean isModelQuantized;
    // Config values.
    private int inputSize;
    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] numDetections;

    private ByteBuffer imgData;

    private Interpreter tfLite;


    // input image dimensions for the Inception Model
    private int DIM_IMG_SIZE_X = 300;
    private int DIM_IMG_SIZE_Y = 300;
    private int DIM_PIXEL_SIZE = 3;

    private GpuDelegate gpuDelegate;

    private Interpreter.Options tfliteOptions;

    protected TFLiteObjectDetectionAPIModel(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize,
            final boolean isQuantized) throws IOException {


        this.create(assetManager,
                modelFilename,
                labelFilename,
                inputSize,
                isQuantized);



    }

    static void setThreads(int threadsSelected){
        NUM_THREADS = threadsSelected;
    }

    static int getThreads(){
        return NUM_THREADS;
    }

    /** Memory-map the model file in Assets. */
    protected MappedByteBuffer loadModelFile(AssetManager assets ,String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename); //assets.openFD
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
    // * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize The size of image input
     * @param isQuantized Boolean representing model is quantized or not
     */
    public void create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize,
            final boolean isQuantized)
            throws IOException {
        //final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

        InputStream labelsInput = null;
//        //String actualFilename = labelFilename.split("file:///android_asset/")[1];
//        System.out.println("HERE ------------------------->" + labelFilename);
        labelsInput = assetManager.open(labelFilename); //actualFilename



        BufferedReader br = null;
        try {
        // ContextInstance.getAssets().open(labelFilename))
        br = new BufferedReader(new InputStreamReader(labelsInput));

        } catch (Exception e) {
            throw new IOException(e);
        }
        String line;
        while ((line = br.readLine()) != null) {
            //LOGGER.w(line);
            this.labels.add(line);
        }
        br.close();

        this.inputSize = inputSize;

        try {
            gpuDelegate = new GpuDelegate();

            tfliteOptions = (new Interpreter.Options()).addDelegate(gpuDelegate);

            this.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), tfliteOptions);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        this.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
//        this.imgData = ByteBuffer.allocateDirect(1 * this.inputSize * this.inputSize * 3 * numBytesPerChannel);
        System.out.println("isQuantized ---------------> "+ isQuantized);
        if(isQuantized){

            imgData =
                    ByteBuffer.allocateDirect(
                            DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        } else {
            imgData =
                    ByteBuffer.allocateDirect(
                            4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        }

        this.imgData.order(ByteOrder.nativeOrder());


        this.intValues = new int[this.inputSize * this.inputSize];
        System.out.println("THIS IS NUMBER OF THREADS SET: " + NUM_THREADS);
        this.tfLite.setNumThreads(NUM_THREADS);
//        this.tfLite.setUseNNAPI(true);
        this.outputLocations = new float[1][NUM_DETECTIONS][4];
        this.outputClasses = new float[1][NUM_DETECTIONS];
        this.outputScores = new float[1][NUM_DETECTIONS];
        this.numDetections = new float[1];
        //return this;
    }

    // resizes bitmap to given dimensions
    private Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }

    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float

                if (this.isModelQuantized) {

                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) (val & 0xFF));
                } else {
                    imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }

            }
        }
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap_orig, int frameWidth, int frameHeight) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");



        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        Bitmap bitmap = null;
        // resize the bitmap to the required input size to the CNN
        bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
        // convert bitmap to byte array
        convertBitmapToByteBuffer(bitmap);

//        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//        imgData.rewind();
//        for (int i = 0; i < inputSize; ++i) {
//            for (int j = 0; j < inputSize; ++j) {
//                int pixelValue = intValues[i * inputSize + j];
//                if (isModelQuantized) {
//                    // Quantized model
//                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
//                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
//                    imgData.put((byte) (pixelValue & 0xFF));
//                } else { // Float model
//                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//                }
//            }
//        }


        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        outputLocations = new float[1][NUM_DETECTIONS][4];
        outputClasses = new float[1][NUM_DETECTIONS];
        outputScores = new float[1][NUM_DETECTIONS];
        numDetections = new float[1];

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap); //tImage.getBuffer(), probabilityBuffer.getBuffer()

        Trace.endSection();

        gpuDelegate.close();


        // Show the best detections.
        // after scaling them back to the input size.

        float[] outputNumDetections = (float[])outputMap.get(3);
        final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
        for (int i = 0; i < NUM_DETECTIONS; ++i) {
            System.out.println("THS IS THE INDEX: "+ i);
            System.out.println("THS IS THE detections: "+ (int)outputNumDetections[0]);

            if ((i == (int)outputNumDetections[0]-1) || ((int)outputNumDetections[0] == 0)){
                break;
            }

            final RectF detection =
                    new RectF(
                            outputLocations[0][i][1] * frameWidth, // * inputsize
                            outputLocations[0][i][0] * frameHeight,
                            outputLocations[0][i][3] * frameWidth,
                            outputLocations[0][i][2] * frameHeight);
            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            int labelOffset = 1;

            //System.out.println("THS IS THE INDEX: "+ i);
            //System.out.println("THS IS THE LABEL.GET(class[0][index]): "+ labels.get((int) outputClasses[0][i]) + ", score: " + outputScores[0][i]);

            recognitions.add(
                    new Recognition(
                            "" + i,
                            labels.get((int) outputClasses[0][i]),// + labelOffset),
                            outputScores[0][i],
                            detection));
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }



    //@Override
    //public void enableStatLogging(final boolean logStats) {}

   // @Override
   // public String getStatString() {
   //     return "";
   // }
    //

    // @Override
   // public void close() {}

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }
}