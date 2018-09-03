var restify = require('restify');
var builder = require('botbuilder');
var numeral = require('numeral');

var MongoClient = require('mongodb').MongoClient
    , assert = require('assert');


var db_url = process.env.MONGODB
var api_url = process.env.API_URL


var findCompany = function (db, company, callback) {
    // Get the documents collection
    var collection = db.collection('companies');
    // Find some documents
    console.log("Looking for " + company.toUpperCase())
    collection.find({ "Symbol": company.toUpperCase()}).toArray(function (err, docs) {
        assert.equal(err, null);
        console.log("Found the following records");
        console.log(docs)
        if (docs.length == 1) {
            callback(docs)
        } else {

            collection.find({ "Name": { $regex: company, $options: 'i' } }).toArray(function (err, docs) {
                console.info("Found " + docs.length)
                callback(docs)
            });
        }

    });
}

function createHeroCard(session, company) {
    console.log(company['Name']);

    return new builder.HeroCard(session, company)
        .title(company['Name'])
        .subtitle(company['Symbol'])
        .text('Listed on the ' + company['Exchange'] + ', ' + company['Name'] + ' is classed in the ' + company['Sector'] + ' sector.  It currently has a market capitlisation of ' + numeral(company['MarketCap']).format('($0.00a)')  + ' and the last known price was $' + companies[i]['LastSale'])
        .images([
            builder.CardImage.create(session, api_url + '/plot/' + company['Symbol'])
            ])
            .buttons([
            builder.CardAction.openUrl(session, 'https://finance.yahoo.com/quote/' + company['Symbol'], 'View on Yahoo! Finance')
        ]);
}

function createHeroCardsMulti(session, companies) {
    var cards = []

    for (var i = 0; i < companies.length; i++) {
        console.info(companies[i]['Name'])

        cards.push(new builder.HeroCard(session, companies[i])
            .title(companies[i]['Name'])
            .subtitle("Symbol: " + companies[i]['Symbol'] + ", Exchange: " + companies[i]['Exchange'])
            .text('Listed on the ' + companies[i]['Exchange'] + ', ' + companies[i]['Name'] + ' is classed in the ' + companies[i]['Sector'] + ' sector.  It currently has a market capitlisation of ' + numeral(companies[i]['MarketCap']).format('($0.00a)') + ' and the last known price was $' + companies[i]['LastSale'])
            .tap(builder.CardAction.imBack(session, companies[i]['Symbol']))
        )
    }

    return cards
}



// Setup Restify Server
var server = restify.createServer();
server.listen(process.env.port || process.env.PORT || 3978, function () {
    console.log('%s listening to %s', server.name, server.url);
});

// Create chat connector for communicating with the Bot Framework Service
var connector = new builder.ChatConnector({
    appId: process.env.MICROSOFT_APP_ID,
    appPassword: process.env.MICROSOFT_APP_PASSWORD
});

// Listen for messages from users 
server.post('/api/messages', connector.listen());

var bot = new builder.UniversalBot(connector);



// Add dialog to handle 'Patterns' button click - not used
bot.dialog('patternsButtonClick',
    function (session, args, next) {
        session.send("Would look up patterns.");
        session.endDialog();
    }
).triggerAction({ matches: /(Patterns)\s.*/i });


// Add dialog to handle 'Add' button click - Not used
bot.dialog('addButtonClick',
    function (session, args, next) {
        session.send("Would add to storage.");
        session.endDialog();

    }
).triggerAction({ matches: /(Add)\s.*/i });


bot.dialog('/', [
    function (session, results) {
        MongoClient.connect(db_url, function (err, db) {
            assert.equal(null, err);
            console.log("Connected successfully to server");

            findCompany(db, session.message.text, function (docs) {

                if (docs.length == 1) {
                    var card = createHeroCard(session, docs[0]);
                    var msg = new builder.Message(session).addAttachment(card);
                    //session.userData.instrument = docs[0]['Symbol'];
                } else if (docs.length > 1) {
                    var cards = createHeroCardsMulti(session, docs);
                    var msg = new builder.Message(session).attachmentLayout(builder.AttachmentLayout.carousel).attachments(cards);
                } else if (docs.length == 0) {
                    session.send("I can't find a ticker or Company by that name.")
                }


                session.send(msg);

                db.close();

            });
        });

    }
]);


bot.on('conversationUpdate', function (activity) {
    // when user joins conversation, send instructions
    if (activity.membersAdded) {
        activity.membersAdded.forEach(function (identity) {
            if (identity.id === activity.address.bot.id) {
                var reply = new builder.Message()
                    .address(activity.address)
                    .text("Hello there! This is a demonstration of setting up a Bot on Microsoft Bot Framework using financial information from Quandl, Azure Functions, and Open Source on Azure.");
                bot.send(reply);
                
                var reply2 = new builder.Message()
                    .address(activity.address)
                    .text("**This Bot is not here to give financial advice, and any Technical Analysis through the 'Patterns' whould not form any investment decisions.**");
                bot.send(reply2);
            
                var reply3 = new builder.Message()
                    .address(activity.address)
                    .text("You can find the walkthough and documentation here: http://bit.ly/2rIozGX - Start your journey by asking me about a company or ticker listed on NASDAQ, AMEX or NYSE exchanges (ex: MSFT or Microsoft)...");
                bot.send(reply3);

                
            }
        });
    }
});