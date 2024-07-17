#include <stdio.h>
#include <string.h>

// Define database structure
struct CyberDatabase {
    char username[50];
    char password[50];
    int rank;
    int experience;
};

int main() {
    // Create a sample database with 3 entries
    struct CyberDatabase database[3] = {
        {""Neo"", ""redpill"", 8, 1432},
        {""Trinity"", ""theone"", 9, 1987},
        {""Morpheus"", ""followthewhiterabbit"", 7, 1289}
    };

    // Prompt user for their username and password
    char user[50];
    char pass[50];
    printf(""Welcome to the Cyber Database. Please enter your credentials to proceed:\n"");
    printf(""Username: ""); 
    scanf(""%s"", user); 
    scanf(""%s"", pass);
    printf(""\n"");

    // Check if user exists in database and password matches
    int user_found = 0;
    int user_rank;
    for (int i = 0; i < 3; i++) {
        if (strcmp(database[i].username, user) == 0 && strcmp(database[i].password, pass) == 0) {
            user_found = 1;
            user_rank = database[i].rank;
            printf(""Welcome back, %s. Your current rank is %d and you have %d experience points.\n"", user, user_rank, database[i].experience);
            break;
        }
    }

    // If user exists, prompt for query command
    if (user_found) {
        printf(""\nWhat would you like to query?\n"");
        printf(""1. Users with a rank of 7 or higher.\n"");
        printf(""2. Users with more than 1500 experience points.\n"");
        printf(""3. Users with a rank of 9 or higher and more than 2000 experience points.\n"");
        int query_num;
        scanf(""%d"", &query_num);
        printf(""\n"");

        // Perform the corresponding query
        switch(query_num) {
            case 1:
                printf(""Users with a rank of 7 or higher:\n"");
                for (int i = 0; i < 3; i++) {
                    if (database[i].rank >= 7) {
                        printf(""%s\n"", database[i].username);
                    }
                }
                break;
            case 2:
                printf(""Users with more than 1500 experience points:\n"");
                for (int i = 0; i < 3; i++) {
                    if (database[i].experience > 1500) {
                        printf(""%s\n"", database[i].username);
                    }
                }
                break;
            case 3:
                printf(""Users with a rank of 9 or higher and more than 2000 experience points:\n"");
                for (int i = 0; i < 3; i++) {
                    if (database[i].rank >= 9 && database[i].experience > 2000) {
                        printf(""%s\n"", database[i].username);
                    }
                }
        }
    } else {
        printf(""Invalid credentials. Terminating program.\n"");
    }

    return 0;
}