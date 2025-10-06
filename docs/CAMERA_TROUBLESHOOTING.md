# 🔧 Camera Troubleshooting Guide

## Common Camera Issues & Solutions

This guide helps resolve camera access problems in the waste classification web application.

---

## ❌ Error: "Requested device not found"

This is the most common camera error. Here are solutions:

### Solution 1: Check Camera Connection

**For Built-in Cameras (Laptops):**
- Camera should be enabled in Device Manager (Windows)
- Check System Preferences → Security & Privacy → Camera (Mac)

**For External Webcams:**
- Ensure USB cable is properly connected
- Try a different USB port
- Check if camera light turns on

**Test Your Camera:**
1. Open Windows Camera app (Windows)
2. Open Photo Booth (Mac)
3. If camera doesn't work there, it's a system issue

### Solution 2: Close Other Applications

Cameras can only be used by one application at a time.

**Close these if running:**
- Zoom, Microsoft Teams, Skype, Google Meet
- OBS Studio, Streamlabs
- Any video recording software
- Other browser tabs using camera
- Discord, Slack (with video)

**How to Check:**
- Windows: Task Manager → See what's running
- Mac: Activity Monitor → Search for "camera"

### Solution 3: Restart Browser

**Complete Browser Restart:**
1. Close ALL browser windows (not just tabs)
2. Wait 5 seconds
3. Reopen browser
4. Try camera again

**Clear Browser Cache:**
- Chrome: Ctrl+Shift+Delete → Clear data
- Firefox: Ctrl+Shift+Delete → Clear cache
- Edge: Ctrl+Shift+Delete → Clear browsing data

### Solution 4: Check Browser Permissions

**Chrome/Edge:**
1. Click the lock icon (🔒) in address bar
2. Click "Site settings"
3. Find "Camera" 
4. Select "Allow"
5. Refresh the page

**Firefox:**
1. Click the lock icon (🔒) in address bar
2. Click "Connection secure" → More information
3. Go to "Permissions" tab
4. Find "Use the Camera"
5. Uncheck "Use default" and check "Allow"

**Safari (Mac):**
1. Safari → Preferences → Websites
2. Click "Camera" in left sidebar
3. Find your site and select "Allow"

### Solution 5: Enable Camera in System Settings

**Windows 10/11:**
1. Settings → Privacy & Security → Camera
2. Turn ON "Allow apps to access your camera"
3. Turn ON "Allow desktop apps to access your camera"
4. Scroll down and enable for your browser

**Mac:**
1. System Preferences → Security & Privacy
2. Click "Privacy" tab
3. Select "Camera" from left menu
4. Check the box next to your browser

**Linux:**
1. Check camera with: `ls /dev/video*`
2. Test with: `cheese` or `guvcview`
3. Grant browser permission if prompted

### Solution 6: Update Browser

Old browsers may have camera compatibility issues.

**Check Version:**
- Chrome: Help → About Google Chrome
- Firefox: Help → About Firefox
- Edge: Help → About Microsoft Edge

**Update to Latest:**
- Chrome: Auto-updates when you restart
- Firefox: Help → About Firefox → Update
- Edge: Settings → About → Update

### Solution 7: Try Different Browser

If one browser doesn't work, try another:

| Browser | Camera Support |
|---------|---------------|
| Chrome | ✅ Excellent |
| Edge | ✅ Excellent |
| Firefox | ✅ Good |
| Safari | ✅ Good |
| Opera | ✅ Good |
| IE 11 | ❌ Not supported |

### Solution 8: Use Camera Selector

If you have multiple cameras:

1. Open the camera modal
2. Look for "Select Camera" dropdown
3. Try each camera option
4. One should work!

### Solution 9: Test with Simple Page

Create a test HTML file to verify camera works:

```html
<!DOCTYPE html>
<html>
<body>
    <h1>Camera Test</h1>
    <video id="video" width="640" height="480" autoplay></video>
    <script>
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                document.getElementById('video').srcObject = stream;
                alert('Camera works!');
            })
            .catch(err => alert('Error: ' + err.message));
    </script>
</body>
</html>
```

Save as `camera-test.html` and open in browser.

### Solution 10: Check Antivirus/Firewall

Some security software blocks camera access:

**Common Culprits:**
- Avast, AVG, Norton, McAfee
- Windows Defender
- Third-party firewalls

**Fix:**
1. Open your security software
2. Find "Camera Protection" or "Webcam Protection"
3. Add browser to allowed list
4. Temporarily disable and test

---

## ⚠️ Error: "Permission Denied"

### Solution: Grant Browser Permission

**First Time:**
- Browser will ask for permission
- Click "Allow" or "Always Allow"
- Don't select "Block"

**If Already Blocked:**

**Chrome/Edge:**
1. Visit: `chrome://settings/content/camera`
2. Find your site under "Blocked"
3. Click trash icon to remove
4. Refresh page, allow when asked

**Firefox:**
1. Click lock icon → More information
2. Permissions tab
3. Camera → Clear and allow again

**Safari:**
1. Safari → Settings for This Website
2. Camera → Allow

---

## ⚠️ Error: "Camera in use"

### Solution: NotReadableError

This means another application is using the camera.

**Quick Fix:**
1. Close all other applications
2. Restart browser completely
3. Try again

**Force Close (Windows):**
1. Task Manager (Ctrl+Shift+Esc)
2. Find processes using camera
3. End task
4. Try again

**Force Close (Mac):**
1. Activity Monitor
2. Search "camera" or "video"
3. Force Quit process
4. Try again

---

## 🐛 Still Not Working?

### Advanced Troubleshooting

#### 1. Check Camera Drivers (Windows)

```
Device Manager → Cameras/Imaging Devices
→ Right-click camera → Update driver
```

#### 2. Check Camera Firmware

Some webcams need firmware updates from manufacturer's website.

#### 3. Try Safe Mode

Restart in Safe Mode to rule out software conflicts.

#### 4. Check Browser Console

1. Press F12 to open Developer Tools
2. Click "Console" tab
3. Look for red error messages
4. Share errors for help

#### 5. Test Camera Properties

```javascript
// In browser console (F12)
navigator.mediaDevices.enumerateDevices()
    .then(devices => console.log(devices));
```

Should show your camera devices. If empty, system issue.

---

## 📱 Mobile Troubleshooting

### iOS (iPhone/iPad)

**Camera Not Working:**
1. Settings → Safari → Camera
2. Select "Allow"
3. Settings → Privacy → Camera
4. Enable Safari

**Wrong Camera Opens:**
- iOS automatically chooses camera
- Back camera preferred for rear-facing mode
- May default to front if only one available

### Android

**Camera Not Working:**
1. Settings → Apps → Browser → Permissions
2. Enable Camera
3. Clear browser cache/data
4. Try Chrome or Firefox

**Wrong Camera:**
- Android should use back camera
- Some phones default to front
- Try camera selector dropdown

---

## 🔍 Diagnostic Checklist

Work through this checklist:

- [ ] Camera works in other apps (Camera app, Zoom, etc.)
- [ ] No other apps using camera currently
- [ ] Browser is up to date
- [ ] Cleared browser cache and cookies
- [ ] Browser has camera permission (system + site)
- [ ] Antivirus not blocking camera
- [ ] Tried different browser
- [ ] Restarted computer
- [ ] Camera drivers updated (Windows)
- [ ] USB camera properly connected
- [ ] Tested on different USB port (external camera)

If all checked and still not working → **System/Hardware issue**

---

## 💡 Prevention Tips

**To Avoid Camera Issues:**

1. **Keep browser updated** - Auto-update enabled
2. **Close camera apps** - When done with Zoom/Teams
3. **Don't block permission** - Click "Allow" first time
4. **Use modern browser** - Chrome, Edge, Firefox latest
5. **Check system settings** - Camera enabled in privacy
6. **Regular restart** - Restart browser occasionally
7. **Use HTTPS** - Required for production (except localhost)

---

## 🆘 Getting Help

### Information to Provide

When asking for help, include:

1. **Error message** - Exact text
2. **Browser** - Name and version
3. **OS** - Windows 11, macOS, etc.
4. **Camera type** - Built-in or external
5. **Console errors** - From F12 → Console
6. **What you tried** - From this guide

### Where to Get Help

- GitHub Issues
- Stack Overflow (tag: [webcam], [getusermedia])
- Browser support forums
- Your IT department (work computers)

---

## 🔄 Alternative: File Upload

If camera still doesn't work, users can:

1. Take photo with phone camera app
2. Save the photo
3. Use "Click to upload" feature instead
4. Drag & drop the saved photo

Both methods work equally well! 📸

---

## ✅ Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Device not found | Close other apps, restart browser |
| Permission denied | Allow in browser settings |
| Camera in use | Close Zoom, Teams, etc. |
| Not working at all | Try different browser |
| Wrong camera opens | Use camera selector dropdown |
| Blurry/dark | Check lighting, camera settings |

---

## 📞 Common Questions

**Q: Why does it work in Zoom but not here?**
A: Zoom may be using camera. Close Zoom completely and try again.

**Q: Can I use my phone camera for desktop site?**
A: Yes! Open site on phone, use camera feature there, or upload photo.

**Q: Do I need to install anything?**
A: No! Camera works in browser, no installation needed.

**Q: Is my camera privacy protected?**
A: Yes! Only you control when camera activates. Photos sent only to your server.

**Q: Why is my external webcam not detected?**
A: Check USB connection, try different port, install manufacturer drivers.

---

**Need more help? Check browser console (F12) for specific error messages!** 🔧

---

## 🎯 Most Common Solution

**95% of camera issues are fixed by:**

1. ✅ Closing Zoom/Teams/Skype
2. ✅ Restarting browser completely
3. ✅ Allowing camera permission
4. ✅ Trying a different browser

**Try these four steps first!** 🚀
