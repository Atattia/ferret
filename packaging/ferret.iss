; Inno Setup script for Ferret (Windows installer)
; Build with: iscc packaging\ferret.iss
; Requires Inno Setup 6: https://jrsoftware.org/isinfo.php

#define MyAppName      "Ferret"
#define MyAppVersion   GetEnv("FERRET_VERSION")
#define MyAppPublisher "Mahmoud Yousry"
#define MyAppExeName   "ferret.exe"
#define MyAppURL       "https://github.com/mahmoudyousry/ferret"

[Setup]
AppId={{6B3A2F1C-4E5D-4A8B-9C0D-1E2F3A4B5C6D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=ferret_{#MyAppVersion}_windows_setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "startup";     Description: "Start Ferret automatically when I log in"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
Source: "dist\ferret\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";                    Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}";          Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";              Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}";              Filename: "{app}\{#MyAppExeName}"; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Ferret now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "taskkill"; Parameters: "/f /im ferret.exe"; Flags: runhidden; RunOnceId: "KillFerret"
