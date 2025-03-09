# Face Database Structure

This directory should contain subdirectories for each known individual, with their face images inside.

Example structure:
```
data/known_faces/
├── person1_name/
│   ├── image1.jpg
│   ├── image2.jpg
├── person2_name/
│   ├── image1.jpg
│   └── image2.jpg
```

Place at least 2-3 clear face images per person in their respective directories. The images should:
- Show the person's face clearly
- Have good lighting
- Be front-facing
- Be in JPG or PNG format

After adding face images, the SDK will be able to recognize these individuals.
