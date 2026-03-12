' run_hidden.vbs — Launch PowerShell script in a completely hidden window
' Usage: wscript.exe run_hidden.vbs "<script.ps1>" [args...]
'
' Task Scheduler + powershell.exe always creates a brief console flash.
' This VBScript wrapper eliminates the popup entirely by using
' WScript.Shell.Run with window-style 0 (SW_HIDE).
'
' 168# register_tasks / 180# popup fix

Option Explicit

Dim shell, fso, scriptPath, args, cmd, i

Set shell = CreateObject("WScript.Shell")
Set fso   = CreateObject("Scripting.FileSystemObject")

If WScript.Arguments.Count < 1 Then
    WScript.Echo "Usage: wscript.exe run_hidden.vbs <script.ps1> [args...]"
    WScript.Quit 1
End If

scriptPath = WScript.Arguments(0)

' Resolve to absolute path if relative
If Not fso.FileExists(scriptPath) Then
    ' Try relative to the VBS script's own directory
    Dim vbsDir
    vbsDir = fso.GetParentFolderName(WScript.ScriptFullName)
    If fso.FileExists(fso.BuildPath(vbsDir, scriptPath)) Then
        scriptPath = fso.BuildPath(vbsDir, scriptPath)
    End If
End If

' Collect remaining arguments
args = ""
If WScript.Arguments.Count > 1 Then
    For i = 1 To WScript.Arguments.Count - 1
        args = args & " " & WScript.Arguments(i)
    Next
End If

cmd = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptPath & """" & args

' Run hidden (0 = SW_HIDE), wait for completion (True)
shell.Run cmd, 0, True
