package com.example.trafficsigndetection;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.SurfaceView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.PopupMenu;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import android.content.res.AssetManager;
import android.widget.ToggleButton;

// extends AppCompatActivity
public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    // camera
    private CameraBridgeViewBase cameraBridgeViewBase;
    private BaseLoaderCallback baseLoaderCallback;

    // ---------------
    // tlite
    // ---------------
    private boolean startTensorflow = false;
    private boolean firstTimeTensorflow = false;
    private AssetManager assetManager;
    private final String modelFilename = "signs_mobilenet.tflite";
    private final String labelFilename = "labels.txt";
    private final int inputSize = 300;
    private final boolean isQuantized = true;
    private TFLiteObjectDetectionAPIModel model;
    private List<Classifier.Recognition> recognitions;
    private Mat frame, frameT;
    private FPSCounter fpsCounter;
    private Integer FPS;
    private int frameWidth;
    private int frameHeight;


    // button names and recover states
    String toggleButton_title;
    ImageButton options;
    boolean showFPS = false;
    boolean makeSound = false;
    String showFPS_title;
    String makeSound_title;
    MediaPlayer mp;
    int threadsSelected;
    ToggleButton toggleButton;

    // sound variables
    AudioManager am;
    int volume_level1;
    int maxVolume;
    float log1;
    String signDetected = null;
    int count = 0;




    public void setDetectionState(View Button) {

        boolean checked = ((ToggleButton)Button).isChecked();
        if (checked){
            toggleButton_title = "STOP";
//            textView.setText(textView_title);


        }
        else {
            toggleButton_title = "START";
//            textView.setText(textView_title);

        }

        if (startTensorflow == false) {

            startTensorflow = true;
            //if (firstTimeTensorflow == false) {
                //firstTimeTensorflow = true;
                assetManager = this.getAssets();
                try {
                    model = new TFLiteObjectDetectionAPIModel(assetManager, modelFilename, labelFilename, inputSize, isQuantized);
                } catch (IOException e) {
                    Log.e("tfliteSupport", "Error initializing model", e);
                }
            //}
        } else {
            startTensorflow = false;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) { //When application launches

        super.onCreate(savedInstanceState);

//        this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);



        // start/stop button
        //textView = (TextView)findViewById(R.id.toggle_button_1);
        toggleButton = (ToggleButton)findViewById(R.id.toggle_button_1);
//            textView.setText("START");

        // initialize sound

        am = (AudioManager) getSystemService(AUDIO_SERVICE);
        volume_level1= am.getStreamVolume(AudioManager.STREAM_RING);
        maxVolume=am.getStreamMaxVolume(AudioManager.STREAM_RING);
        log1=(float)(1-Math.log(maxVolume-volume_level1)/Math.log(maxVolume));
        mp  = MediaPlayer.create(this, R.raw.tone_beep); //getApplicationContext()
        mp.setAudioStreamType(AudioManager.STREAM_MUSIC);
        mp.setVolume(log1,log1);



        // options button
        options = (ImageButton)findViewById(R.id.options);
        options.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Creating the instance of PopupMenu
                PopupMenu popup = new PopupMenu(MainActivity.this, options);
                //Inflating the Popup using xml file
                popup.getMenuInflater()
                        .inflate(R.menu.popup_menu, popup.getMenu());
                Menu m = popup.getMenu();

                // change title
                if (showFPS){
                    showFPS_title = "FPS: On";
                    m.findItem(R.id.fps).setTitle(showFPS_title);

                }
                else{
                    showFPS_title = "FPS: Off";
                    m.findItem(R.id.fps).setTitle(showFPS_title);

                }
                if (makeSound){
                    makeSound_title = "Sound: On";
                    m.findItem(R.id.sound).setTitle(makeSound_title);
                }
                else{
                    makeSound_title = "Sound: Off";
                    m.findItem(R.id.sound).setTitle(makeSound_title);
                }
                // showing threads as selected
                System.out.println("threadsSelected a: "+ threadsSelected);
                System.out.println("threadsSelected b: "+ threadsSelected);
                System.out.println("threadsSelected c: "+ threadsSelected);
                System.out.println("threadsSelected d: "+ threadsSelected);
                System.out.println("TFLiteObjectDetectionAPIModel e: "+ TFLiteObjectDetectionAPIModel.getThreads());
                System.out.println("TFLiteObjectDetectionAPIModel f: "+ TFLiteObjectDetectionAPIModel.getThreads());

                if (TFLiteObjectDetectionAPIModel.getThreads() == 0){
                    threadsSelected=1;
                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);
                }
                if (TFLiteObjectDetectionAPIModel.getThreads() == 1) {
                    m.findItem(R.id.One).setChecked(true);
                }
                else if (TFLiteObjectDetectionAPIModel.getThreads() == 2) {
                    m.findItem(R.id.Two).setChecked(true);
                }
                else if (TFLiteObjectDetectionAPIModel.getThreads() == 3) {
                    m.findItem(R.id.Three).setChecked(true);
                }
                else if (TFLiteObjectDetectionAPIModel.getThreads() == 4) {
                    m.findItem(R.id.Four).setChecked(true);
                }





                //registering popup with OnMenuItemClickListener
                popup.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {


                    public boolean onMenuItemClick(MenuItem item) {
//                        Toast.makeText(
//                                MainActivity.this,
//                                "You Clicked : " + item.getTitle(),
//                                Toast.LENGTH_SHORT
//                        ).show();


                        switch (item.getItemId()) {
                            case R.id.fps:

                                if (showFPS == false){
                                    showFPS = true;
                                   }
                                else{
                                    showFPS = false;
                                }


                                return true;

                            case R.id.sound:
                                if (makeSound == false){
                                    makeSound = true;
                                }
                                else{
                                    makeSound = false;
                                }

                                return true;

                            case R.id.One:
                                System.out.println("CLICKED ON ONE");
                                threadsSelected = 1;


                                if (startTensorflow) {
                                    System.out.println("THREADS PASSED" + threadsSelected);
                                    model.setNumThreads(threadsSelected);
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);

                                }
                                else{
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);
                                }
                                return true;
                            case R.id.Two:
                                System.out.println("CLICKED ON TWO");
                                threadsSelected = 2;


                                if (startTensorflow) {
                                    System.out.println("THREADS PASSED" + threadsSelected);
                                    model.setNumThreads(threadsSelected);
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);

                                }
                                else{
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);
                                }
                                return true;
                            case R.id.Three:
                                System.out.println("CLICKED ON THREE");
                                threadsSelected = 3;


                                if (startTensorflow) {
                                    System.out.println("THREADS PASSED" + threadsSelected);
                                    model.setNumThreads(threadsSelected);
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);

                                }
                                else{
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);
                                }
                                return true;
                            case R.id.Four:
                                System.out.println("CLICKED ON FOUR");
                                threadsSelected = 4;


                                if (startTensorflow) {
                                    model.setNumThreads(threadsSelected);
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);
                                    System.out.println("THREADS PASSED" + threadsSelected);
                                }
                                else{
                                    TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);
                                }
                                return true;
                            default:
                                return false;
                        }




                        //return true;

            }
        });
                popup.show(); //showing popup menu (popup menu had different libraries)
            }
        }); //closing the setOnClickListener method

        // camera
        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        // Check if works properly
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                switch (status) {
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }




    // 20-30 fps per second on average
    // Mat stores images in matrix objects, because java do not have numpy
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frame = inputFrame.rgba(); // right now app is a camera


        //----------------------------------------------------
        //              T E N S O R F L O W
        //----------------------------------------------------


        if (startTensorflow == true) {
            Bitmap bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(frame, bitmap);
            //System.out.println("BEFORE RECOGNITIONS ---------------------------");

            // -----skip frame if null -----
            System.out.println("FRAME: " + frame);
            System.out.println("BITMAP: " + bitmap.getByteCount());

                if (frame != null)
                try
                {
                recognitions = model.recognizeImage(bitmap, frameWidth, frameHeight);
                //System.out.println(recognitions);
//            [[0] no overtaking trucks (94,5%) RectF(196.30403, 157.14839, 205.82161, 172.23038), [1] , , , ,]
                float confThreshold = 0.3f;
                List<Integer> clsIds = new ArrayList<>(); // msut be Integer
                List<Float> confs = new ArrayList<>();
                List<Rect> rects = new ArrayList<>();
                List<String> titles = new ArrayList<>();

                for (int i = 0; i < recognitions.size(); ++i) {
                    float confidence = recognitions.get(i).getConfidence();
                    if (confidence > confThreshold) {
//                    System.out.println(recognitions.get(i).getLocation());
//                    System.out.println("FRAME cols" + frame.cols());
//                    System.out.println("FRAME rows" + frame.rows());

                        int left = (int) (recognitions.get(i).getLocation().left);// / inputSize * frame.cols());
                        int top = (int) (recognitions.get(i).getLocation().top);// / inputSize * frame.rows());
                        int right = (int) (recognitions.get(i).getLocation().right);// / inputSize * frame.cols());
                        int bottom = (int) (recognitions.get(i).getLocation().bottom); // / inputSize * frame.rows());
                        int width = (int) recognitions.get(i).getLocation().width();
                        int height = (int) recognitions.get(i).getLocation().height();




                        try {
                            clsIds.add(Integer.parseInt(recognitions.get(i).getId())); // must be int
                        } catch (NumberFormatException e) {
                            System.out.println("Cannot convert Id to integer");
                        }
                        confs.add((float) confidence);
                        titles.add(recognitions.get(i).getTitle());
                        rects.add(new Rect(left, top, width, height));
//                    rects.add(new Rect(left, top, right, bottom));
//                    System.out.println("add RECT --------->" + new Rect(left, top, right, bottom));
                    }
                }
                int ArrayLength = confs.size();
                if (ArrayLength >= 1) {
//                // Apply non-maximum suppression procedure.
                    float nmsThresh = 0.2f;
                    MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs)); // confs
                    Rect[] boxesArray = rects.toArray(new Rect[0]); // Rect not RectF
                    MatOfRect boxes = new MatOfRect(boxesArray);
                    MatOfInt indices = new MatOfInt();
                    Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
//                // Draw result boxes:
                    int[] ind = indices.toArray();
                    for (int i = 0; i < ind.length; ++i) {
                        int idx = ind[i];
                        Rect box = boxesArray[idx];
                        int idGuy = clsIds.get(idx);
                        float conf = confs.get(idx);
                        int intConf = (int) (conf * 100);


                        // beep sound

                        if (makeSound==true && intConf>90) {
                            mp.start();
                        }


                        signDetected = titles.get(idGuy);

                        // put text
                        Imgproc.putText(frame, titles.get(idGuy) + " " + intConf + "%", box.tl(), Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 0), 2);
                        System.out.println("TOP LEFT " + box.tl() + " \nBOTTOM RIGHT " + box.br());
                        Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);

                    }
                }

            } catch(java.lang.NullPointerException exception){
                    System.out.println(exception + " --> recognizeImage(android.graphics.Bitmap, int, int)' on a null object reference");
                }
        }

        if (showFPS == true){
        fpsCounter.logFrame();
        FPS = (int)fpsCounter.getFPS();
        Imgproc.putText(frame, "FPS: " + FPS, new Point(10, 50),Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 0), 2);
        }


        return frame;
    }



    @Override
    public void onCameraViewStarted(int width, int height) {
        frame = new Mat(height, width, CvType.CV_8UC4);
        frameWidth = width;
        frameHeight = height;
        fpsCounter = new FPSCounter();

        if (startTensorflow == true) {
            assetManager = this.getAssets();
            try {
                model = new TFLiteObjectDetectionAPIModel(assetManager, modelFilename, labelFilename, inputSize, isQuantized);
            } catch (IOException e) {
                Log.e("tfliteSupport", "Error initializing model onCameraViewStarted", e);
            }

        }
    }

    @Override
    public void onCameraViewStopped() {
        frame.release();
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Returns true, if not true then something bad
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "There is a problem.", Toast.LENGTH_SHORT).show();
        } else {// If ok then successs
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() { // If screen turns off
        super.onPause();
        if (cameraBridgeViewBase != null) { // If it is working
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() { // When app is terminated
        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
        if(mp!=null){
            mp.release();
        }
    }
    @Override
    public void onSaveInstanceState(Bundle savedInstanceState) {
        super.onSaveInstanceState(savedInstanceState);
        // Save UI state changes to the savedInstanceState.
        // This bundle will be passed to onCreate if the process is
        // killed and restarted.
        savedInstanceState.putString("toggleButton_title_key", toggleButton_title);
        savedInstanceState.putBoolean("showFPS_key", showFPS);
        savedInstanceState.putString("showFPS_title_key", showFPS_title);
        savedInstanceState.putBoolean("makeSound_key", makeSound);
        savedInstanceState.putString("makeSound_title_key", makeSound_title);
        savedInstanceState.putInt("threadsSelected_key", threadsSelected);
        // etc.
    }

    @Override
    public void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        // Restore UI state from the savedInstanceState.
        // This bundle has also been passed to onCreate.

        toggleButton_title = savedInstanceState.getString("toggleButton_title_key");
        showFPS = savedInstanceState.getBoolean("showFPS_key");
        showFPS_title = savedInstanceState.getString("showFPS_title_key");
        makeSound = savedInstanceState.getBoolean("makeSound_key");
        makeSound_title = savedInstanceState.getString("makeSound_title");
        threadsSelected = savedInstanceState.getInt("threadsSelected_key");
        if (toggleButton_title == "STOP"){
            toggleButton.setChecked(false);
        }

        if (startTensorflow){
            model.setNumThreads(threadsSelected);
        }else {
            TFLiteObjectDetectionAPIModel.setThreads(threadsSelected);
        }
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_UP:
                am.adjustStreamVolume(
                        AudioManager.STREAM_MUSIC,
                        AudioManager.ADJUST_RAISE,
                        AudioManager.FLAG_PLAY_SOUND | AudioManager.FLAG_SHOW_UI);
                return true;
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                am.adjustStreamVolume(
                        AudioManager.STREAM_MUSIC,
                        AudioManager.ADJUST_LOWER,
                        AudioManager.FLAG_PLAY_SOUND | AudioManager.FLAG_SHOW_UI);
                return true;
            default:
                break;
        }
        return super.onKeyUp(keyCode, event);
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {


        switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_UP:
                am.adjustStreamVolume(
                        AudioManager.STREAM_MUSIC,
                        AudioManager.ADJUST_RAISE,
                        AudioManager.FLAG_PLAY_SOUND | AudioManager.FLAG_SHOW_UI);
                return true;
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                am.adjustStreamVolume(
                        AudioManager.STREAM_MUSIC,
                        AudioManager.ADJUST_LOWER,
                        AudioManager.FLAG_PLAY_SOUND | AudioManager.FLAG_SHOW_UI);
                return true;
            default:
                break;
        }
        return super.onKeyDown(keyCode, event);
    }


}




