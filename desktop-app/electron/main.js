const { app, BrowserWindow } = require('electron');
const path = require('path');

// Enable GPU hardware acceleration for RTX 4060
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('enable-webgl');
app.commandLine.appendSwitch('enable-accelerated-2d-canvas');

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: '#07090e',
    title: 'PREDATOR-NET — AI Neural Firewall & Antivirus Cockpit',
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
    },
  });

  const devUrl = 'http://localhost:5173';
  mainWindow.loadURL(devUrl).catch(() => {
    // If dev server not yet active, fallback to dist
    const prodPath = path.join(__dirname, '../dist/index.html');
    mainWindow.loadFile(prodPath);
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
