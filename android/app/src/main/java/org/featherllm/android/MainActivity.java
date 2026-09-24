package org.featherllm.android;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public final class MainActivity extends Activity {
    static {
        System.loadLibrary("featherllm_android");
    }

    private static native String nativeRuntimeVersion();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView status = new TextView(this);
        status.setText(nativeRuntimeVersion());
        status.setTextSize(18.0f);
        status.setPadding(32, 32, 32, 32);
        setContentView(status);
    }
}
