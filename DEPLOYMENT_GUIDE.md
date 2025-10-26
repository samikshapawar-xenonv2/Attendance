# DEPLOYMENT GUIDE - Face Recognition Attendance System

## ⚠️ IMPORTANT: This App Cannot Run on Vercel

This application uses:
- Camera/Webcam access
- OpenCV (computer vision)
- Face recognition with live video streaming
- Long-running processes

**Vercel is designed for:**
- Static websites
- Serverless API functions (max 10-second execution)
- Next.js apps

**This app needs a traditional server that can:**
- Access hardware (camera)
- Run continuously
- Handle video streaming

---

## ✅ RECOMMENDED DEPLOYMENT OPTIONS

### **Option 1: Render.com (Best for Flask + OpenCV)**

**Why Render?**
- ✅ Free tier available
- ✅ Supports Python/Flask perfectly
- ✅ Can handle OpenCV and face-recognition libraries
- ✅ Automatic deployments from GitHub
- ✅ Custom domains

**Steps to Deploy:**

1. **Create Account**
   - Go to https://render.com
   - Sign up with your GitHub account

2. **Prepare Your Repository**
   - Add this to your repo (already done): `requirements.txt`
   - Create `render.yaml` (see below)

3. **Create Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repo: `Amardighe1/Attendance`
   - Configure:
     ```
     Name: attendance-system
     Environment: Python 3
     Build Command: pip install -r requirements.txt
     Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
     ```

4. **Add Environment Variables**
   - Go to "Environment" tab
   - Add:
     ```
     SUPABASE_URL = https://qvlcmhsbmsdxiipfktjj.supabase.co
     SUPABASE_KEY = your_key_here
     ENCODINGS_FILE = face_encodings.pkl
     ```

5. **Deploy!**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

**Note:** Camera access will only work if you access the deployed site from a device with a camera (laptop/phone).

---

### **Option 2: Railway.app (Easiest Setup)**

1. Go to https://railway.app
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repo
5. Add environment variables
6. Deploy!

**Auto-detects Python and installs dependencies automatically.**

---

### **Option 3: For Local/Campus Network Access**

**Best Option for Classroom Use:**

Run the app on a dedicated laptop/computer in the classroom:

```powershell
# Make it accessible to other devices on the same network
python app.py
```

Then update `app.py` to listen on all network interfaces:

**Change line 527 in app.py from:**
```python
app.run(debug=True)
```

**To:**
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

Now anyone on your local network can access it at:
`http://YOUR_COMPUTER_IP:5000`

Find your IP with: `ipconfig` (look for IPv4 Address)

---

### **Option 4: PythonAnywhere (Python-Specific Hosting)**

1. Go to https://www.pythonanywhere.com
2. Create free account
3. Upload your code
4. Configure Flask app
5. Done!

---

## 📦 ADDITIONAL FILES NEEDED FOR DEPLOYMENT

### 1. Create `gunicorn` requirement (for production server)

Add to `requirements.txt`:
```
gunicorn==21.2.0
```

### 2. Create `Procfile` (for some platforms)

```
web: gunicorn app:app
```

### 3. Update `app.py` for production

Change the last lines to:
```python
if __name__ == '__main__':
    if load_model():
        # Production: use host='0.0.0.0'
        # Local: use host='127.0.0.1'
        app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 🎯 RECOMMENDED SOLUTION FOR YOUR USE CASE

**For College/Campus Attendance:**

### **Local Server Setup (Best for Classroom)**
1. Use a dedicated laptop/desktop in each classroom
2. Run the Flask app on that computer
3. Connect to college WiFi
4. Other devices access via: `http://teacher-laptop-ip:5000`

**Advantages:**
- ✅ Direct camera access
- ✅ Fast face recognition
- ✅ No internet dependency
- ✅ Privacy (data stays local)
- ✅ No hosting costs

**Disadvantages:**
- ❌ Computer must stay on
- ❌ Only works on local network

---

## 🚫 Why Not Vercel?

**Vercel Limitations:**
- Max 10-second function execution
- No persistent processes
- No hardware access (camera)
- Designed for static sites and APIs
- Not suitable for OpenCV/video streaming

**Error you got:**
```
ModuleNotFoundError: No module named 'distutils'
```
This happens because numpy compilation fails in serverless environment.

---

## 📝 QUICK DECISION GUIDE

**Choose LOCAL if:**
- ✅ Single classroom/building
- ✅ Have dedicated computer
- ✅ Want maximum privacy
- ✅ Need fast face recognition

**Choose RENDER/RAILWAY if:**
- ✅ Need internet access
- ✅ Multiple locations
- ✅ Want automatic backups
- ✅ Professional deployment

---

## 🔧 NEED HELP DEPLOYING?

Let me know which option you want to pursue and I can:
1. Create the necessary config files
2. Update app.py for production
3. Guide you through the deployment process
