[Setup]
AppName=mu DNG Converter
AppVersion={#AppVersion}
AppPublisher=mu-files
AppPublisherURL=https://github.com/mu-files/mu-image
AppSupportURL=https://github.com/mu-files/mu-image/issues
DefaultDirName={autopf}\mu DNG Converter
DefaultGroupName=mu DNG Converter
OutputDir=.
OutputBaseFilename=mu-dng-converter-windows-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\mu-dng-converter.exe
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\mu DNG Converter"; Filename: "{app}\mu-dng-converter.exe"
Name: "{group}\{cm:UninstallProgram,mu DNG Converter}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\mu DNG Converter"; Filename: "{app}\mu-dng-converter.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\mu-dng-converter.exe"; Description: "{cm:LaunchProgram,mu DNG Converter}"; Flags: nowait postinstall skipifsilent
