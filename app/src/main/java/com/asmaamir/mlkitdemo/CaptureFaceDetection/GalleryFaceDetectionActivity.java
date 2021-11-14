package com.asmaamir.mlkitdemo.CaptureFaceDetection;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.asmaamir.mlkitdemo.R;
import com.example.kloadingspin.KLoadingSpin;
import com.google.android.gms.tasks.OnCanceledListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionPoint;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceContour;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark;

import java.io.IOException;
import java.text.MessageFormat;
import java.util.List;
import java.util.Objects;

public class GalleryFaceDetectionActivity extends AppCompatActivity {
    public static final int REQUEST_CODE_PERMISSION = 111;
    public static final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.READ_EXTERNAL_STORAGE"};
    private static final String TAG = "PickActivity";
    private static final int PICK_IMAGE_CODE = 100;
    private ImageView imageView;
    private ImageView imageViewCanvas;
    private FirebaseVisionImage image;
    private Bitmap bitmap;
    private Canvas canvas;
    private Paint dotPaint, linePaint, rectPaint;
    private TextView textView;
    private TextView faceCountTextView;
    private Context context;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery_face_detection);
        context = this;
        if (allPermissionsGranted()) {
            initViews();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSION);
        }

    }

    private void initViews() {
        imageView = findViewById(R.id.img_view_pick);
        ImageButton imageButton = findViewById(R.id.img_btn_pick);
        imageViewCanvas = findViewById(R.id.img_view_pick_canvas);
        textView = findViewById(R.id.tv_props);
        faceCountTextView = findViewById(R.id.tv_count);
        imageButton.setOnClickListener(v -> pickImage());
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE_CODE) {
            if (data != null) {
                imageView.setImageURI(data.getData());
                showLoading();
                try {
                    image = FirebaseVisionImage.fromFilePath(context, Objects.requireNonNull(data.getData()));
                    initDetector(image);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void showLoading() {
        KLoadingSpin a = findViewById(R.id.KLoadingSpin);
        a.startAnimation();
        a.setIsVisible(true);
    }

    private void hideLoading() {
        KLoadingSpin a = findViewById(R.id.KLoadingSpin);
        a.stopAnimation();
    }

    private void initDetector(FirebaseVisionImage image) {

        initDrawingUtils();
        FirebaseVisionFaceDetectorOptions detectorOptions = new FirebaseVisionFaceDetectorOptions
                .Builder()
                .setMinFaceSize(.1F)
                .setPerformanceMode(FirebaseVisionFaceDetectorOptions.ACCURATE)
                .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                .build();
        FirebaseVisionFaceDetector faceDetector = FirebaseVision
                .getInstance()
                .getVisionFaceDetector(detectorOptions);
        faceDetector
                .detectInImage(image)
                .addOnCanceledListener((OnCanceledListener) () -> {
                    hideLoading();
                })
                .addOnSuccessListener(firebaseVisionFaces -> {
                    hideLoading();
                    if (!firebaseVisionFaces.isEmpty()) {

                        processFaces(firebaseVisionFaces);
                    } else {
                        faceCountTextView.setText(R.string.no_face_text);
                        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY);
                        Log.i(TAG, "No faces");
                    }
                }).addOnFailureListener(e -> {
                    hideLoading();
                    Log.i(TAG, e.toString());
                    Toast.makeText(context, e.toString(), Toast.LENGTH_LONG).show();
                }
        );
    }

    private void processFaces(List<FirebaseVisionFace> faces) {
        faceCountTextView.setText(MessageFormat.format(getString(R.string.face_count_text), faces.size()));
        int smileFaceCount = 0, sadFaceCount = 0, noCareFaceCount = 0;
        for (FirebaseVisionFace face : faces) {

            Rect bounds = face.getBoundingBox();
            canvas.drawRect(bounds, rectPaint);
            FaceExpression faceExpression = getProps(face);
            switch (faceExpression) {
                case Sad:
                    sadFaceCount++;
                    break;
                case Happy:
                    smileFaceCount++;
                    break;
                case Normal:
                    noCareFaceCount++;
                    break;
                case NotDetected:
                    break;
            }
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EAR));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EAR));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EYE));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EYE));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.MOUTH_BOTTOM));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.MOUTH_LEFT));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.MOUTH_RIGHT));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.LEFT_CHEEK));
            drawLandMark(face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_CHEEK));
            drawContours(face.getContour(FirebaseVisionFaceContour.FACE).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.LEFT_EYEBROW_BOTTOM).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.RIGHT_EYEBROW_BOTTOM).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.LEFT_EYE).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.RIGHT_EYE).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.LEFT_EYEBROW_TOP).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.RIGHT_EYEBROW_TOP).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.LOWER_LIP_BOTTOM).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.LOWER_LIP_TOP).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.UPPER_LIP_BOTTOM).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.UPPER_LIP_TOP).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.NOSE_BRIDGE).getPoints());
            drawContours(face.getContour(FirebaseVisionFaceContour.NOSE_BOTTOM).getPoints());
        }
        imageViewCanvas.setImageBitmap(bitmap);

        textView.setText(getFacesPropText(smileFaceCount, sadFaceCount, noCareFaceCount, faces.size()));

    }

    private String getFacesPropText(int smileFaceCount, int sadFaceCount, int noCareFaceCount, int size) {
        String output;
        if (size > 0) {
            output = "Out of " + size + " faces ";
            output += smileFaceCount > 0 ? smileFaceCount + " are happy :)" : " no one is happy :( ";
            output += sadFaceCount > 0 ? sadFaceCount + " are sad :( " : " no one is sad :)";
            output += noCareFaceCount > 0 ? " and " + noCareFaceCount + " ppl don't care " : " and no one is normal here!";
            return output;
        } else {
            return getResources().getString(R.string.no_face_text);
        }

    }

    private FaceExpression getProps(FirebaseVisionFace face) {
        float smileProb = 0, rightEyeOpenProb = 0, leftEyeOpenProb = 0;
        FaceExpression faceExpression;
        if (face.getSmilingProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
            smileProb = face.getSmilingProbability();
            faceExpression = smileProb > .6 ? FaceExpression.Happy : smileProb < .4 ? FaceExpression.Sad : FaceExpression.Normal;
        } else {
            faceExpression = FaceExpression.NotDetected;
        }

        if (face.getRightEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
            rightEyeOpenProb = face.getRightEyeOpenProbability();
        }
        if (face.getLeftEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
            leftEyeOpenProb = face.getRightEyeOpenProbability();
        }

        textView.setText(textView.getText()
                + " " + smileProb
                + " " + rightEyeOpenProb
                + " " + leftEyeOpenProb
                + "\n\n"
        );
        return faceExpression;
    }


    private void drawLandMark(FirebaseVisionFaceLandmark landmark) {
        if (landmark != null) {
            canvas.drawCircle(landmark.getPosition().getX(), landmark.getPosition().getY(), 10, dotPaint);
        }
    }

    private void drawContours(List<FirebaseVisionPoint> points) {
        int counter = 0;
        for (FirebaseVisionPoint point : points) {
            if (counter != points.size() - 1) {
                canvas.drawLine(point.getX(),
                        point.getY(),
                        points.get(counter + 1).getX(),
                        points.get(counter + 1).getY(),
                        linePaint);
            } else {
                canvas.drawLine(point.getX(),
                        point.getY(),
                        points.get(0).getX(),
                        points.get(0).getY(),
                        linePaint);
            }
            counter++;
            canvas.drawCircle(point.getX(), point.getY(), 6, dotPaint);
        }
    }

    private void initDrawingUtils() {
        bitmap = Bitmap.createBitmap(image.getBitmap().getWidth(),
                image.getBitmap().getHeight(),
                Bitmap.Config.ARGB_8888);
        if (canvas != null)
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY);

        canvas = new Canvas(bitmap);
        dotPaint = new Paint();
        dotPaint.setColor(Color.RED);
        dotPaint.setStyle(Paint.Style.FILL);
        dotPaint.setStrokeWidth(2f);
        dotPaint.setAntiAlias(true);
        linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(2f);
        rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(4f);
        rectPaint.setColor(Color.BLUE);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSION) {
            if (allPermissionsGranted()) {
                initViews();
            } else {
                Toast.makeText(context,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
}