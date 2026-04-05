"""
Test file to validate chromedriver and selenium installation.
Tests basic chromedriver functionality based on the pattern used in layout_check.py
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path


def test_chromedriver_basic():
    """Test basic chromedriver initialization and functionality."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        
        print("✓ Selenium imported successfully")
        
        # Attempt to find chromedriver in the project path
        chromedriver_locations = [
            # "chromedriver.exe",  # Windows, in current directory
            # "./chromedriver.exe",
            # str(Path(__file__).parent / "chromedriver.exe"),
            "./chrome/chromedriver-linux64/chromedriver"
        ]
        
        webdriver_path = None
        for location in chromedriver_locations:
            if os.path.exists(location):
                webdriver_path = location
                print(f"✓ Found chromedriver at: {os.path.abspath(location)}")
                break
        
        if webdriver_path is None:
            print("✗ Chromedriver not found in expected locations")
            print(f"  Searched: {chromedriver_locations}")
            return False
        
        # Initialize Chrome in headless mode
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")
        options.binary_location = "./chrome/chrome-linux64/chrome"
        
        print("✓ Chrome options configured (headless mode)")
        
        try:
            service = Service(webdriver_path)
            driver = webdriver.Chrome(service=service, options=options)
            print("✓ Chrome webdriver initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Chrome webdriver: {e}")
            return False
        
        # Test 1: Navigate to a file and execute JavaScript
        try:
            # Create a temporary HTML file
            temp_html = f"""
            <html>
            <head><title>Test Page</title></head>
            <body>
                <div id="test_element">Hello from Selenium!</div>
                <svg id="axes_1" viewBox="0 0 100 100">
                    <rect x="10" y="10" width="80" height="80" fill="blue"/>
                </svg>
            </body>
            </html>
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                f.write(temp_html)
                temp_file = f.name
            
            print(f"✓ Created temporary test HTML file: {temp_file}")
            
            # Navigate to the file
            driver.get(f"file://{temp_file}")
            print("✓ Successfully navigated to local HTML file")
            
            # Test JavaScript execution
            result = driver.execute_script("""
                let element = document.getElementById('test_element');
                return element.textContent;
            """)
            
            if result == "Hello from Selenium!":
                print(f"✓ JavaScript execution successful: '{result}'")
            else:
                print(f"✗ JavaScript execution returned unexpected result: {result}")
                return False
            
            # Test SVG element access (similar to layout_check.py)
            svg_result = driver.execute_script("""
                let svgElement = document.querySelector('#axes_1');
                if (svgElement) {
                    let bbox = svgElement.getBBox();
                    return {
                        x: bbox.x,
                        y: bbox.y,
                        width: bbox.width,
                        height: bbox.height
                    };
                }
                return null;
            """)
            
            if svg_result:
                print(f"✓ SVG element access successful: {svg_result}")
            else:
                print("✗ Failed to access SVG element")
                return False
            
            # Cleanup
            os.remove(temp_file)
            print("✓ Temporary test file cleaned up")
            
        except Exception as e:
            print(f"✗ Error during webdriver operations: {e}")
            driver.close()
            return False
        finally:
            driver.close()
            print("✓ Webdriver closed successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import selenium: {e}")
        print("  Please install selenium: pip install selenium")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_layout_check_compatibility():
    """Test that chromedriver works with the layout_check.py pattern."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        
        # Find chromedriver
        chromedriver_locations = [
            "./chrome/chromedriver-linux64/chromedriver"
        ]
        
        webdriver_path = None
        for location in chromedriver_locations:
            if os.path.exists(location):
                webdriver_path = location
                break
        
        if webdriver_path is None:
            print("✗ Chromedriver not found for layout_check compatibility test")
            return False
        
        # Create a simple SVG similar to what layout_check processes
        svg_content = """
        <svg viewBox="0 0 100 100" id="test_svg">
            <g id="axes_1">
                <text id="text_1" x="10" y="20">Label 1</text>
                <text id="text_2" x="30" y="40">Label 2</text>
            </g>
        </svg>
        """
        
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")
        options.binary_location = "./chrome/chrome-linux64/chrome"
        
        service = Service(webdriver_path)
        driver = webdriver.Chrome(service=service, options=options)
        
        # Create temporary SVG file
        current_directory = os.getcwd()
        file_path = f"{current_directory}/temp_{uuid.uuid1()}.svg"
        
        with open(file_path, "w") as svg_file:
            svg_file.write(svg_content)
        
        # Load and test
        driver.get(f"file://{file_path}")
        
        # Execute a pattern similar to layout_check.py
        test_script = """
        const findTextElements = (parentElement) => {
            let index = 1;
            let count = 0;
            while (true) {
                let textElement = parentElement.querySelector('#text_' + index);
                if (textElement) {
                    count++;
                    index++;
                } else {
                    break;
                }
            }
            return count;
        }
        
        let svgElement = document.querySelector('#axes_1');
        return findTextElements(svgElement);
        """
        
        result = driver.execute_script(test_script)
        
        driver.close()
        os.remove(file_path)
        
        if result == 2:
            print(f"✓ layout_check.py compatibility test passed: Found {result} text elements")
            return True
        else:
            print(f"✗ layout_check.py compatibility test failed: Expected 2 elements, got {result}")
            return False
            
    except Exception as e:
        print(f"✗ layout_check compatibility test error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Chromedriver and Selenium Installation")
    print("=" * 60)
    
    print("\n[Test 1] Basic Chromedriver Functionality")
    print("-" * 60)
    test1_passed = test_chromedriver_basic()
    
    print("\n[Test 2] layout_check.py Compatibility")
    print("-" * 60)
    test2_passed = test_layout_check_compatibility()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✓ All tests passed! Chromedriver is properly configured.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
