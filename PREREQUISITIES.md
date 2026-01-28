# Required software 

## Table of Contents
- [Common Prerequisites](#common-prerequisites)
- [UiPath Cloud Configuration](#uipath-cloud-configuration)
- [Anthropic Configuration](#anthropic-configuration)
- [macOS Prerequisites](#macos-prerequisites)
- [Windows Prerequisites](#windows-prerequisites)

## Common Prerequisites

### Required Software
**Git**
   - For version control and cloning the repository
   - Check version: `git --version`

## UiPath Cloud Configuration

### 1. UiPath Account Setup

**Create UiPath Account**:
   - Sign up at: https://cloud.uipath.com/tpenlabs/AgenticILT

## macOS Prerequisites

### System Requirements
- macOS 11 (Big Sur) or later

### macOS-Specific Configuration

- **Terminal Access**: Ensure Terminal has Full Disk Access in System Preferences > Security & Privacy

- **SSL Certificates**: macOS may require additional certificates for HTTPS requests:
  ```bash
  pip install --upgrade certifi
  ```

  If the SSL Certificates validation still fails with error `https.ConnectError: [SSL: CERTIFY_VERIFY_FAILED]`, the problem could be caused by a proxy, and trusting the proxy CA might help:
  ``` bash
  # example for mitmproxy
  export SSL_CERT_FILE=~/.mitmproxy/mitmproxy-ca-cert.pem
  ```

## Windows Prerequisites

### System Requirements
- Windows 10 version 1903 or later / Windows 11
- PowerShell 5.1 or later

### Windows-Specific Configuration

- **Execution Policy**: May need to allow script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Long Path Support**: Enable for deep directory structures:
  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```

### Python Dependencies
The project uses `pip` for dependency management. Core dependencies include:
- `uipath-langchain>=0.0.106` - UiPath integration for LangChain
- `langchain-openai` - OpenAI GPT integration
- `ipykernel` - Jupyter notebook support
- `jupyter` - Jupyter notebook environment
