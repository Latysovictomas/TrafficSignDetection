package com.example.trafficsigndetection;

public class FPSCounter {
    private long lastFrame = System.nanoTime();
    private float FPS = 0;

    public void logFrame() {
        long time = (System.nanoTime() - lastFrame);
        FPS = 1/(time/1000000000.0f);
        lastFrame = System.nanoTime();
    }

    public float getFPS(){
        return this.FPS;
    }



}