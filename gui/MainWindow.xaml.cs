using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Wpf.Ui.Controls;
using static System.Runtime.InteropServices.JavaScript.JSType;
using MessageBox = Wpf.Ui.Controls.MessageBox;
using MessageBoxButton = Wpf.Ui.Controls.MessageBoxButton;



namespace FlowersDetecting
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : FluentWindow
    {
        private string selectedImagePath = ""; // Store the selected image path
        private string appFolderPath;
        private HttpClient httpClient;



        public MainWindow()
        {
            DataContext = this;

            Wpf.Ui.Appearance.SystemThemeWatcher.Watch(this);
            InitializeComponent();
            appFolderPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "AppData");
            Directory.CreateDirectory(appFolderPath);

            httpClient = new HttpClient();
            
        }





        private async void installPythonButton_Click(object sender, RoutedEventArgs e)
        {
            //verify if python with pip is already installed
            if (File.Exists(appFolderPath + "/python/python.exe") && File.Exists(appFolderPath + "/python/Scripts/pip.exe"))
            {
                var Message = new MessageBox
                {
                    Title = "Python Installed",
                    Content = "Python and pip are installed successfully",
                };
                await Message.ShowDialogAsync();
                return;
            }

            Directory.CreateDirectory(appFolderPath + "/python");

            string downloadPath = System.IO.Path.Combine(appFolderPath, "python/python.zip");
            loadingRing.Visibility = Visibility.Visible;
            progressBar.Visibility = Visibility.Visible;
            progressBar.Value = 0;
            progressBar.IsIndeterminate = false;

            try
            {
                using (var response = await httpClient.GetAsync("https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip", HttpCompletionOption.ResponseHeadersRead))
                {
                    response.EnsureSuccessStatusCode(); // Throw if not 200-299

                    long? totalBytes = response.Content.Headers.ContentLength;

                    using (var contentStream = await response.Content.ReadAsStreamAsync())
                    using (var fileStream = new FileStream(downloadPath, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        var buffer = new byte[8192]; // 8KB buffer
                        long totalBytesRead = 0;

                        int bytesRead;
                        while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) != 0)
                        {
                            await fileStream.WriteAsync(buffer, 0, bytesRead);
                            totalBytesRead += bytesRead;

                            if (totalBytes.HasValue)
                            {
                                progressBar.Value = (double)totalBytesRead / totalBytes.Value * 100;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                var messageBox = new MessageBox
                {
                    Title = "Error",
                    Content = "Error downloading Python:\n" + ex.Message,
                };
                await messageBox.ShowDialogAsync();
            }
            finally
            {
                progressBar.Visibility = Visibility.Collapsed;
                loadingRing.Visibility = Visibility.Collapsed;
            }

            //install pip https://bootstrap.pypa.io/get-pip.py
            try
            {
                using (var response = await httpClient.GetAsync("https://bootstrap.pypa.io/get-pip.py", HttpCompletionOption.ResponseHeadersRead))
                {
                    response.EnsureSuccessStatusCode(); // Throw if not 200-299
                    long? totalBytes = response.Content.Headers.ContentLength;
                    using (var contentStream = await response.Content.ReadAsStreamAsync())
                    using (var fileStream = new FileStream(appFolderPath + "/python/get-pip.py", FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        var buffer = new byte[8192]; // 8KB buffer
                        long totalBytesRead = 0;
                        int bytesRead;
                        while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) != 0)
                        {
                            await fileStream.WriteAsync(buffer, 0, bytesRead);
                            totalBytesRead += bytesRead;

                            if (totalBytes.HasValue)
                            {
                                progressBar.Value = (double)totalBytesRead / totalBytes.Value * 100;
                            }
                        }
                    }
                }
            }
            finally
            {

                ZipFile.ExtractToDirectory(downloadPath, appFolderPath + "/python", true); // Overwrite
                File.Delete(downloadPath);
                //install pip 
                progressBar.Visibility = Visibility.Visible;
                progressBar.IsIndeterminate = true;
                
                await Task.Run(async () =>
                {
                 ProcessStartInfo start = new ProcessStartInfo();
                start.FileName = appFolderPath + "/python/python.exe";
                start.Arguments = appFolderPath + "/python/get-pip.py --no-warn-script-location";
                start.UseShellExecute = false;
                start.CreateNoWindow = true;
                start.RedirectStandardOutput = true; // Capture output
                start.RedirectStandardError = true;  // Capture errors
                start.RedirectStandardInput = true;

                using (Process process = Process.Start(start))
                {
                    // Read output and error streams asynchronously
                    var outputTask = process.StandardOutput.ReadToEndAsync();
                    var errorTask = process.StandardError.ReadToEndAsync();
                    process.WaitForExit();
                    // Retrieve output and error messages (after process exits)
                    string output = await outputTask;
                    string error = await errorTask;
                    // Handle output/errors as needed (e.g., display in a textbox)
                    if (!string.IsNullOrEmpty(error))
                    {
                        // Log or display the error
                        var messageBox1 = new MessageBox
                        {
                            Title = "Error",
                            Content = "Error installing pip:\n" + error,
                        };
                        await messageBox1.ShowDialogAsync();
                    }
                    else if (!string.IsNullOrEmpty(output))
                    {
                        // Optionally display the output (might be verbose)
                        //Dispatcher.Invoke(() => { /* ... */ });
                    }
                }
                //uncoment import site from python312._pth
                string[] lines = File.ReadAllLines(appFolderPath + "/python/python312._pth");
                for (int i = 0; i < lines.Length; i++)
                {
                    if (lines[i].Contains("#import site"))
                    {
                        lines[i] = lines[i].Replace("#import site", "import site");
                    }
                }
                File.WriteAllLines(appFolderPath + "/python/python312._pth", lines);


            });
                progressBar.Visibility = Visibility.Collapsed;
                var messageBox = new MessageBox
                {
                    Title = "Python Installed",
                    Content = "Python and pip are installed successfully",
                };

                await messageBox.ShowDialogAsync();
            }
        }







        private async void imageChooserButton_Click(object sender, RoutedEventArgs e)
        {
            // Create a new file picker
            var picker = new Microsoft.Win32.OpenFileDialog();
            picker.Filter = "Image files (*.jpg, *.jpeg, *.png)|*.jpg;*.jpeg;*.png";
            // Show the file picker so the user can select an image
            var result = picker.ShowDialog();
            if (result == true)
            {
                // Get the image file path
                selectedImagePath = picker.FileName;
                // Display the selected image
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(picker.FileName);
                bitmap.EndInit();
                imagePreview.Source = bitmap;
            }
        }










        // ...

        private async void installDepsButton_Click(object sender, RoutedEventArgs e)
        {
            progressBar.Visibility = Visibility.Visible;
            progressBar.IsIndeterminate = true;
            resultTextBox.Text = ""; // Clear previous output

            try
            {
                await Task.Run(async () =>
                {
                    ProcessStartInfo start = new ProcessStartInfo();
                    start.FileName = appFolderPath + "/python/python.exe";
                    start.Arguments = "-m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121";
                    start.UseShellExecute = false;
                    start.CreateNoWindow = true;
                    start.RedirectStandardOutput = true;
                    start.RedirectStandardError = true;


                    using (Process process = Process.Start(start))
                    {
                        // Read output and error streams asynchronously
                        using (StreamReader reader = process.StandardOutput)
                        {
                            while (!process.StandardOutput.EndOfStream)
                            {
                                string line = await reader.ReadLineAsync();
                                UpdateOutput(line);
                            }
                        }
                        using (StreamReader reader = process.StandardError)
                        {
                            while (!process.StandardError.EndOfStream)
                            {
                                string line = await reader.ReadLineAsync();
                                UpdateOutput(line, isError: true);
                            }
                        }
                        process.WaitForExit();

                        if (process.ExitCode != 0)
                        {
                            // Show error message if pip exits with a non-zero code
                            UpdateOutput($"Pip install failed with exit code {process.ExitCode}", isError: true);
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                UpdateOutput($"An unexpected error occurred: {ex.Message}", isError: true);
            }
            finally
            {
                progressBar.Visibility = Visibility.Collapsed;
            }
        }

        private void UpdateOutput(string text, bool isError = false)
        {
            Dispatcher.Invoke(() =>
            {
                if (isError)
                {
                    resultTextBox.AppendText($"[ERROR] {text}\r\n");
                }
                else
                {
                    resultTextBox.AppendText($"{text}\r\n");
                }
                resultTextBox.ScrollToEnd();
            });
        }













        private async void checkButton_Click(object sender, RoutedEventArgs e)
        {
    

            bool Downloaded = false;
            if (string.IsNullOrEmpty(selectedImagePath))
            {
                // Handle case where no image is selected
                // E.g., display an error message
                System.Diagnostics.Debug.WriteLine("No image selected.");
                return;
            }
            //check if scripts are already downloaded
            if (File.Exists(appFolderPath + "/WPZ_2024-25_NR_10-main/test_labeling.py"))
            {
                Downloaded = true;
                System.Diagnostics.Debug.WriteLine("Files downloaded");

            }
            progressBar.Visibility = Visibility.Visible;
            progressBar.IsIndeterminate = true;
            //download scipts from github project https://github.com/NeroPcStation/WPZ_2024-25_NR_10/archive/refs/heads/main.zip
            if (!Downloaded)
            {
                loadingRing.Visibility = Visibility.Visible;


                try
                {
                    progressBar.Visibility = Visibility.Visible;
                    progressBar.IsIndeterminate = false;
                    progressBar.Value = 0;
                    using (var response = httpClient.GetAsync("https://github.com/NeroPcStation/WPZ_2024-25_NR_10/archive/refs/heads/main.zip", HttpCompletionOption.ResponseHeadersRead))
                    {
                        response.Result.EnsureSuccessStatusCode(); // Throw if not 200-299
                        long? totalBytes = response.Result.Content.Headers.ContentLength;
                        using (var contentStream = response.Result.Content.ReadAsStreamAsync())
                        using (var fileStream = new FileStream(appFolderPath + "/WPZ_2024-25_NR_10-main.zip", FileMode.Create, FileAccess.Write, FileShare.None))
                        {
                            var buffer = new byte[8192]; // 8KB buffer
                            long totalBytesRead = 0;
                            int bytesRead;
                            while ((bytesRead = contentStream.Result.Read(buffer, 0, buffer.Length)) != 0)
                            {
                                fileStream.Write(buffer, 0, bytesRead);
                                totalBytesRead += bytesRead;
                                if (totalBytes.HasValue)
                                {
                                    progressBar.Value = (double)totalBytesRead / totalBytes.Value * 100;
                                }
                            }
                        }




                    }


                }
                catch
                {
                    var messageBox = new MessageBox
                    {
                        Title = "Error",
                        Content = "Error downloading scripts",
                    };
                    await messageBox.ShowDialogAsync();
                    return;
                }
                finally
                {
                    progressBar.Visibility = Visibility.Collapsed;
                    ZipFile.ExtractToDirectory(appFolderPath + "/WPZ_2024-25_NR_10-main.zip", appFolderPath, true); // Overwrite
                    File.Delete(appFolderPath + "/WPZ_2024-25_NR_10-main.zip");
                }
            }




            try
            {

                progressBar.Visibility = Visibility.Visible;
                progressBar.IsIndeterminate = true;
                resultTextBox.Text = "";
                errorTextBox.Text = "";
                await Task.Run(async () =>
                {
                    ProcessStartInfo start = new ProcessStartInfo();
                start.FileName = appFolderPath + "/python/python.exe";
                start.WorkingDirectory = appFolderPath + "/WPZ_2024-25_NR_10-main/";

                start.Arguments = appFolderPath + "/WPZ_2024-25_NR_10-main/test_labeling.py" + " " + selectedImagePath;


                start.UseShellExecute = false;
                start.CreateNoWindow = true;

                start.RedirectStandardOutput = true;
                start.RedirectStandardError = true; // Capture errors



                    using (Process process = Process.Start(start))
                    {
                        // Read output and error streams asynchronously
                        using (StreamReader reader = process.StandardOutput)
                        {
                            while (!process.StandardOutput.EndOfStream)
                            {
                                string line = await reader.ReadLineAsync();
                                UpdateOutput(line);
                            }
                        }
                        using (StreamReader reader = process.StandardError)
                        {
                            while (!process.StandardError.EndOfStream)
                            {
                                string line = await reader.ReadLineAsync();
                                UpdateOutput(line, isError: true);
                            }
                        }
                        process.WaitForExit();

                        if (process.ExitCode != 0)
                        {
                            // Show error message if pip exits with a non-zero code
                            UpdateOutput($"Pip install failed with exit code {process.ExitCode}", isError: true);
                        }
                    }

                });
            }
            catch (Exception ex)
            {
                // Handle exceptions (e.g., Python not found)
                Console.WriteLine($"Error running Python script: {ex.Message}");
                // Sho
            }
            loadingRing.Visibility = Visibility.Collapsed;
            progressBar.Visibility = Visibility.Collapsed;

        }

 

        private void imagePreview_Drop(object sender, DragEventArgs e)
        {

            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                selectedImagePath = files[0];
                // Display the selected image
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(files[0]);
                bitmap.EndInit();
                imagePreview.Source = bitmap;
            }


        }

        private void imagePreview_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Copy; // Indicate that we accept the drop
            }
            else
            {
                e.Effects = DragDropEffects.None; // Indicate that we don't accept the drop
            }

        }
    }
}