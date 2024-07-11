pretty_print() {
  printf "\n%b\n" "$1"
}

if ! command -v brew &> /dev/null; then
  pretty_print "Homebrew is not installed. Installing Homebrew..."
  
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Add Homebrew to the PATH
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
  eval "$(/opt/homebrew/bin/brew shellenv)"
  
  pretty_print "Homebrew installation completed."
else
  pretty_print "Homebrew is already installed."
fi