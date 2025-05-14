[app]

# Title and package info
title = Binance Scanner
package.name = binancescanner
package.domain = org.yourdomain

# Source files
source.dir = .
source.include_exts = py,png,jpg,kv,ttf

# Version
version = 1.0

# Requirements - MUST INCLUDE THESE
requirements = 
    python3,
    kivy==2.1.0,
    pandas,
    numpy,
    aiohttp,
    ta,  # Using ta instead of talib for Android compatibility
    asyncio,
    android

# Android permissions - CRITICAL FOR BACKGROUND OPERATION
android.permissions = 
    INTERNET,
    FOREGROUND_SERVICE,
    WAKE_LOCK,
    RECEIVE_BOOT_COMPLETED,
    POST_NOTIFICATIONS

# Android configuration
android.api = 31
android.minapi = 21
android.ndk = 23b
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = True
android.gradle_dependencies = 'com.android.support:multidex:1.0.3'

# Orientation
orientation = portrait
fullscreen = 0

# Important build settings
android.accept_sdk_license = True
android.wakelock = True  # Required for background operation

# TA-Lib alternative (using 'ta' library)
# Remove if you want to use talib instead (requires manual .so files)