I have added new features to the existing project available on GitHub.
https://github.com/nay-hey/DevHack.git

## Project Setup Instructions

### Prerequisites
1. **Download the project**:
   - Download the project zip file and extract it.

2. **Open the project**:
   - Open the extracted folder using Visual Studio Code.

3. **Install MySQL**:
   - Download and install MySQL on your system.
   - Install the MySQL extension for Visual Studio Code.

### Database Setup
1. **Create a super user**:
   - Run the following command in your terminal, replacing `*username*` with your super user username:
     ```
     mysql -u *username* -p
     ```

2. **Create a new MySQL user** (if required):
   - If you need to create a new MySQL user, run:
     ```sql
     CREATE USER '*username*'@'localhost' IDENTIFIED BY '*password*';
     ```

3. **Create a new database**:
   - To create a new database named `reservations`, run:
     ```sql
     CREATE DATABASE reservations;
     ```
   - If the database already exists, you can check using:
     ```sql
     SHOW DATABASES;
     ```
   - To drop the existing database, run:
     ```sql
     DROP DATABASE reservations;
     ```
   - Alternatively, give a new name to the database and update the `NAME` field in the `DATABASES` setting in `settings.py`.

4. **Grant privileges**:
   - Grant the user access to the database:
     ```sql
     GRANT ALL PRIVILEGES ON reservations.* TO '*username*'@'localhost';
     FLUSH PRIVILEGES;
     ```

5. **Change settings.py in CentralPerk folder**:
   - Go to DATABASES section:
     ```sql
     DATABASES = {
          'default': {
              'ENGINE': 'django.db.backends.mysql',
              'NAME': #'name',
              'HOST' : '127.0.0.1',
              'PORT' : '3306',
              'USER' : #'username',
              'PASSWORD' : #'password',
          }
      }
     ```
   - Edit this DATABASES section accordingly.

### Django Setup
1. **Make migrations**:
   - In the terminal, navigate to your project directory and run:
     ```
     python manage.py makemigrations
     python manage.py migrate
     ```

2. **Run the development server**:
   - Start the Django development server:
     ```
     python manage.py runserver
     ```

3. **Access the website**:
   - Open your web browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### Additional Notes
- Ensure you update the `DATABASES` setting in `settings.py` with the correct database name, user, and password.
