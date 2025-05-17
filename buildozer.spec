[app]

# (str) Title of your application
title = Your Binance Scanner App

# (str) Package name
package.name = binancescanner

# (str) Package domain (needed for android/ios packaging)
package.domain = com.yourdomain

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let buildozer figure it out)
# source.include_exts = py,png,jpg,kv,atlas,txt,json # Add txt if you use symbols.txt

# (list) List of inclusions using pattern matching
# source.include_patterns = assets/*,images/*.png

# (list) Source files to exclude (let buildozer figure it out)
# source.exclude_exts = spec

# (list) List of directory to exclude (let buildozer figure it out)
# source.exclude_dirs = tests,bin,v*

# (str) Application versioning (e.g. 1.0.0)
version = 1.0.0

# (list) Kivy requirements
# Comma-separated list of requirements
# Ensure you have the exact names as on PyPI or Kivy's recipes
requirements = python3,kivy,aiohttp,pandas,numpy,ta,pyjnius

# (str) Custom source folders for requirements
# requirements.source.kivymd = ../../kivymd

# (str) Presplash background color (hex RRGGBB or RRGGBBAA)
# android.presplash_color = #FFFFFF

# (str) Presplash image
presplash.filename = %(source.dir)s/presplash.png

# (str) Icon filename
icon.filename = %(source.dir)s/icon.png

# (str) Supported orientation (landscape, portrait, all)
orientation = portrait

# (list) List of service definitions
# services = Name:entrypoint_of_service.py
# No explicit service declaration needed here as it's started dynamically via AndroidService from main.py.
# The FOREGROUND_SERVICE permission is the key.

#
# Android specific
#

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (string) Android entry point, default is ok for Kivy-based app
android.entrypoint = org.kivy.android.PythonActivity

# (string) Android app theme, default is ok for Kivy-based app
# android.apptheme = "@android:style/Theme.NoTitleBar" # Example for no title bar
android.apptheme = "@android:style/Theme.Material.Light.NoActionBar" # A modern theme

# (list) Permissions
android.permissions =
    INTERNET,
    ACCESS_NETWORK_STATE, # Good practice for checking network connectivity
    FOREGROUND_SERVICE,
    WAKE_LOCK,
    POST_NOTIFICATIONS, # Required for Android 13 (API 33) and above to show notifications
    RECEIVE_BOOT_COMPLETED # NOTE: Requires additional native Android (Java/Kotlin) code (BroadcastReceiver) to actually start your service on boot.

# (int) Android API level to target
# Target Android 13 (API 33) or higher for POST_NOTIFICATIONS to be fully relevant and for modern features
android.api = 33

# (int) Minimum API level
android.minapi = 23 # Android 6.0 (Marshmallow) - balances compatibility and modern features

# (int) Android SDK version to use
android.sdk = 20 # Usually fine, p4a manages this.

# (int) Android NDK version to use
android.ndk = 19c 
# Common stable choice, p4a often defaults or downloads a suitable one.

# (str) Android NDK directory (if not using an auto-downloaded one)
# android.ndk_path = /path/to/your/ndk

# (str) Android SDK directory (if not using an auto-downloaded one)
# android.sdk_path = /path/to/your/sdk

# (bool) Use --enable-androidx for androidx support. Recommended for modern apps.
android.enable_androidx = True

# (list) Android architectures to build for
android.archs = arm64-v8a, armeabi-v7a

# (str) The Android build tools version to use.
# Default is latest. You can specify, e.g., "30.0.3"
# android.build_tools_version = <specific_version>

# (str) Python for android branch to use
# Use 'master' for the latest, or a specific release tag for stability (e.g., '2023.05.21')
p4a.branch = master
# p4a.source_dir = /path/to/your/python-for-android # if you have a local clone

# (bool) If True, then skip trying to update the Android sdk
# This can be useful if you already have a working SDK setup
android.skip_update = False

# (list) Pattern to whitelist for the asset pack
# android.whitelist_asset_dirs = Falsenone

# (bool) Copy libraries to distributable folder.
# android.copy_libs = True # Default is True

# (str) The log level to use when running the app.
# Available levels: 'v', 'd', 'i', 'w', 'e', 'f'. Default is 'i'.
# android.logcat_filters = *:S python:D  # Example: Silence all, show Python debug logs

# (list) Android libraries to add to AndroidManifest.xml
# android.add_libs_arm64_v8a = libs/arm64-v8a/myjni.so
# android.add_libs_armeabi_v7a = libs/armeabi-v7a/myjni.so

# (list) Java classes to add to make .java files from .pyx files
# android.add_src = java_src_directory_relative_to_project_root

# (list) gradle dependencies to add
# android.gradle_dependencies = com.android.support:multidex:1.0.3
# android.gradle_dependencies = androidx.work:work-runtime:2.7.1 # If you were to use WorkManager directly

# (str) Path to a custom AndroidManifest.xml template
# android.manifest.template = /path/to/your/AndroidManifest.tmpl.xml

# (str) Path to a custom build.gradle template
# android.build_gradle.template = /path/to/your/build.tmpl.gradle

# (str) Path to a custom strings.xml template
# android.strings_xml.template = /path/to/your/strings.tmpl.xml

# (str) Android AAB file path for building an app bundle.
# android.aab_path = %(source.dir)s/dist/%(app.name)s-%(app.version)s.aab # Example

#
# Python for android (p4a) specific
#

# (str) p4a directory - DEPRECATED
# p4a.dir =

# (str) The directory in which python-for-android should look for recipes
# p4a.local_recipes = /path/to/your/recipes

# (str) Command-line arguments to pass to p4a
# p4a.extra_args = --debug

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to build GNUMake configuration file
# make_config = %(source.dir)s/.GNUMakeconfig

# (str) Path to build NDK Profile file
# ndk_profile = %(source.dir)s/.ndkprofile

# (str) LLVM version to use with NDK. Valid from NDK 23+.
# If not set, the default LLVM version included with the NDK will be used.
# android.ndk_llvm_version = clang # Example for default clang