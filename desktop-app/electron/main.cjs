const { app, BrowserWindow, shell } = require('electron');
const path = require('path');

// Ensure ELECTRON_RUN_AS_NODE does not leak into child processes
delete process.env.ELECTRON_RUN_AS_NODE;

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
    backgroundColor: '#0B0F17',
    title: 'PREDATOR-NET — AI Neural Firewall & Antivirus Cockpit',
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: false,
      webSecurity: false,
    },
  });

  const prodPath = path.join(__dirname, '../dist/index.html');
  const isDev = process.env.VITE_DEV === 'true';

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173').catch(() => {
      mainWindow.loadFile(prodPath);
    });
  } else {
    mainWindow.loadFile(prodPath).catch((err) => {
      console.error('Error loading dist/index.html:', err);
    });
  }

  // Gracefully show native desktop window once ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Open external links in user's browser, keeping desktop app contained
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('did-fail-load:', errorCode, errorDescription);
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
