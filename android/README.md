# IPDDS for Android

IPDDS means Intelligent Plant Disease Detection System. The app checks Potato and Tomato leaves offline using the embedded crop and 13-class disease models, with matching treatment information. AI results are estimates and should be confirmed before treatment.

## Open and build

Open the `android` folder in Android Studio, allow Gradle sync to finish, then choose **Build > Build APK(s)**.

The debug APK is written to:

`android/app/build/outputs/apk/debug/app-debug.apk`

You can install it on a connected Android phone with:

```powershell
.\gradlew.bat installDebug
```

Or install the APK directly:

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

The project targets Android 7.0 or newer. Choose a saved image with the system photo picker or take a new one with the phone's camera app. It does not request photo-library, camera, or internet permissions.

The interface follows the device's system light or dark appearance. Use the sidebar to open the leaf checker, About, and Settings. About shows model details and reported accuracy; Settings can remember the last plant choice. Its launcher icon uses a leaf mark on a dark background.

## How image checks work

Choose or take a close-up photo, keep **Auto-detect** selected or choose Potato/Tomato manually, then tap **Check leaf**. A local color, coverage, texture, and model-confidence check rejects blank or visibly unrelated images before showing a diagnosis. At least 20% of the center quarter of the image must contain leaf-like color. This is a heuristic, not a separately trained leaf detector; other objects can still fool it, so important results need confirmation. Large photos are downsampled before processing. The selected image is converted to RGB, resized to 128 by 128, and passed to both embedded TensorFlow Lite models as float values from 0 to 255. The model's own rescaling layer performs the same normalization used during training.

Auto-detection uses the crop model's Potato/Tomato output and a conservative Tomato confidence threshold. Selecting a crop manually restricts the disease result to that crop's labels. Disease names and treatments are loaded from the same class metadata used by the web app.

The app runs inference locally. Its result is a guide and should not replace professional diagnosis.
